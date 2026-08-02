from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

import tools.orin_deploy.fcooper_orin_finalize as finalizer_module
from tools.orin_deploy.fcooper_orin_finalize import (
    ARMS,
    FinalizationError,
    finalize,
)


SCOPE = "post_scatter_backbone_shrinker"
DISPLAY = {
    "original_default": "Original/default",
    "compression_only": "Compression only",
    "schedule_only": "Schedule only",
    "compress_then_tune": "Compress -> Tune",
    "joint_fp16_control": "Joint FP16 control",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def canonical_ids() -> list[str]:
    return [f"opv2v-test:{index}:cav-{index}" for index in range(2170)]


def id_sequence_sha(ids: list[str]) -> str:
    encoded = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_samples(path: Path, count: int = 1500, latency: float = 2.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "cuda_event_ms"])
        writer.writerows((index, latency) for index in range(count))


def create_complete_tree(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "results"
    manifest_path = root / "00_source_audit/canonical_manifest.json"
    arms: dict[str, dict] = {}
    widths = {
        "original_default": [64, 128, 256, 128, 256],
        "compression_only": [32, 64, 64, 32, 64],
        "schedule_only": [64, 128, 256, 128, 256],
        "compress_then_tune": [64, 64, 64, 32, 64],
        "joint_fp16_control": [32, 32, 64, 32, 64],
    }
    for arm in ARMS:
        source = root / "00_source_audit/source" / arm
        checkpoint, config, onnx = (
            source / "model.pth",
            source / "config.yaml",
            source / "model.onnx",
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{arm}".encode())
        config.write_bytes(f"config-{arm}".encode())
        onnx.write_bytes(f"onnx-{arm}".encode())
        is_native = arm == "original_default"
        arms[arm] = {
            "runtime": "native" if is_native else "trt",
            "widths": widths[arm],
            "q_mode": "fp32" if arm in {"original_default", "schedule_only"} else "fp16",
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": digest(checkpoint),
            "config_path": str(config.resolve()),
            "config_sha256": digest(config),
            "onnx_path": str(onnx.resolve()),
            "onnx_sha256": digest(onnx),
            "input_shape": [5, 64, 512, 512],
            "builder_policy": None if is_native else "trt85_default",
            "builder_optimization_level": None,
        }
    arms["schedule_only"]["strict_fp32"] = True
    arms["joint_fp16_control"]["joint_cross_precision_control"] = True
    manifest = {
        "status": "ready",
        "scope": SCOPE,
        "input_shape": [5, 64, 512, 512],
        "arms": arms,
        "global_identities": {
            "contracts/opv2v_test_manifest_fresh.txt": {
                "sha256": "e70afc9c82d29d405d86aa75fc6651a0fec77e88b825934c6464650f3416aafb"
            }
        },
    }
    write_json(manifest_path, manifest)
    manifest_sha = digest(manifest_path)

    heldout_input = root / "01_numeric_gate/heldout_inputs.npy"
    heldout_input.parent.mkdir(parents=True, exist_ok=True)
    heldout_input.write_bytes(b"synthetic-heldout-npy")
    heldout_manifest = root / "01_numeric_gate/heldout_manifest.json"
    write_json(
        heldout_manifest,
        {
            "status": "ready",
            "purpose": "heldout_numeric_only_not_calibration",
            "calibration_eligible": False,
            "scope": SCOPE,
            "dataset_samples": 2170,
            "tensor_contract": {
                "shape": [3, 5, 64, 512, 512],
                "dtype": "float32",
                "engine_agent_batch": 5,
            },
            "items": [
                {
                    "stable_id": f"opv2v-test:{index}:cav-{index}",
                    "tensor_sha256": f"{index + 1:064x}",
                    "agent_count": 1,
                }
                for index in range(3)
            ],
            "npy_sha256": digest(heldout_input),
            "opv2v_test_manifest_sha256": manifest["global_identities"][
                "contracts/opv2v_test_manifest_fresh.txt"
            ]["sha256"],
        },
    )
    write_json(
        heldout_input.with_suffix(".heldout.json"),
        {
            "input_sha256": digest(heldout_input),
            "capture_provenance_sha256": digest(heldout_manifest),
            "opv2v_test_manifest_sha256": manifest["global_identities"][
                "contracts/opv2v_test_manifest_fresh.txt"
            ]["sha256"],
            "purpose": "heldout_numeric_only_not_calibration",
            "calibration_eligible": False,
            "scope": SCOPE,
        },
    )
    heldout_sha = digest(heldout_input)
    capture_sha = digest(heldout_manifest)

    for position, arm in enumerate(ARMS):
        spec = arms[arm]
        is_native = arm == "original_default"
        engine_sha = None
        if is_native:
            write_json(
                root / f"01_numeric_gate/{arm}/reference.json",
                {
                    "status": "complete",
                    "kind": "reference",
                    "arm": arm,
                    "scope": SCOPE,
                    "input_sha256": heldout_sha,
                    "checkpoint_sha256": spec["checkpoint_sha256"],
                    "config_sha256": spec["config_sha256"],
                    "outputs": {
                        f"item_{index}": {
                            "finite": True,
                            "shape": [5, 256, 256, 256],
                            "dtype": "float32",
                        }
                        for index in range(3)
                    },
                },
            )
        else:
            engine_dir = root / f"02_engines/{arm}"
            engine = engine_dir / "model.engine"
            cache = engine_dir / "timing.cache"
            inspector = engine_dir / "inspector.json"
            engine.parent.mkdir(parents=True, exist_ok=True)
            engine.write_bytes(f"engine-{arm}".encode())
            cache.write_bytes(f"cache-{arm}".encode())
            engine_sha = digest(engine)
            layers = (
                [{"name": "conv", "precision": "FP32", "compute_precision": "FP32"}]
                if arm == "schedule_only"
                else [{"name": "conv", "precision": "FP16"}]
            )
            write_json(inspector, {"layers": layers, "unsupported_fields": []})
            write_json(
                engine_dir / "build.json",
                {
                    "status": "built",
                    "scope": SCOPE,
                    "arm": arm,
                    "manifest_sha256": manifest_sha,
                    "engine": str(engine.resolve()),
                    "timing_cache": str(cache.resolve()),
                    "engine_sha256": engine_sha,
                    "timing_cache_sha256": digest(cache),
                    "checkpoint_sha256": spec["checkpoint_sha256"],
                    "config_sha256": spec["config_sha256"],
                    "onnx_sha256": spec["onnx_sha256"],
                    "platform": ["Linux", "aarch64", "NVIDIA Jetson Orin"],
                    "tensorrt_version": "8.5.2.2",
                    "build_arguments": {
                        "builder_policy": spec["builder_policy"],
                        "builder_optimization_level": spec["builder_optimization_level"],
                        "q_mode": spec["q_mode"],
                    },
                    "h800_engine_or_cache_read": False,
                    "calibration_cache_read": False,
                },
            )
            write_json(
                root / f"01_numeric_gate/{arm}/gate.json",
                {
                    "status": "pass",
                    "manifest_sha256": manifest_sha,
                    "input_sha256": heldout_sha,
                    "capture_provenance_sha256": capture_sha,
                    "opv2v_test_manifest_sha256": manifest["global_identities"][
                        "contracts/opv2v_test_manifest_fresh.txt"
                    ]["sha256"],
                    "checkpoint_sha256": spec["checkpoint_sha256"],
                    "config_sha256": spec["config_sha256"],
                    "onnx_sha256": spec["onnx_sha256"],
                    "engine_sha256": engine_sha,
                    "tolerances": (
                        {
                            "cosine_min": 0.99999,
                            "nrmse_max": 0.01,
                            "mae_max": 0.001,
                            "max_abs_max": 0.05,
                        }
                        if spec["q_mode"] == "fp32"
                        else {
                            "cosine_min": 0.999,
                            "nrmse_max": 0.05,
                            "mae_max": 0.05,
                            "max_abs_max": 0.5,
                        }
                    ),
                    "items": {
                        f"item_{index}": {
                            "cosine": 0.999999,
                            "nRMSE": 0.00001,
                            "MAE": 0.00001,
                            "max_abs": 0.0001,
                            "finite_ratio": 1.0,
                            "shape": [5, 64, 256, 256],
                            "reference_dtype": "float32",
                            "actual_dtype": "float32",
                            "reference_min": 0.0,
                            "reference_max": 1.0,
                            "reference_zero_ratio": 0.1,
                            "actual_min": 0.0,
                            "actual_max": 1.0,
                            "actual_zero_ratio": 0.1,
                        }
                        for index in range(3)
                    },
                },
            )

        latency_dir = root / f"03_latency/{arm}"
        samples = latency_dir / "samples.csv"
        power_log = root / f"04_energy/raw_tegrastats/{arm}.log"
        write_samples(samples)
        power_log.parent.mkdir(parents=True, exist_ok=True)
        power_log.write_text("t0 VIN_SYS_5V0 5000mW VDD_GPU_SOC 9000mW\n")
        write_json(
            latency_dir / "report.json",
            {
                "status": "complete",
                "arm": arm,
                "scope": SCOPE,
                "protocol": {"warmup": 20, "iters": 300, "repeat": 5, "sample_count": 1500},
                "sample_count": 1500,
                "pooled": {
                    "median_ms": 2.0,
                    "p90_ms": 2.0,
                    "p99_ms": 2.0,
                    "mean_ms": 2.0,
                },
                "rails": {
                    "VIN_SYS_5V0": {
                        "watts": [5.0],
                        "timestamps": ["t0"],
                        "sample_count": 1,
                    },
                    "VDD_GPU_SOC": {
                        "watts": [9.0],
                        "timestamps": ["t0"],
                        "sample_count": 1,
                    },
                },
                "energy_j": 0.01,
                "raw_power_log_sha256": digest(power_log),
                "input_sha256": heldout_sha,
                "capture_provenance_sha256": capture_sha,
                "checkpoint_sha256": spec["checkpoint_sha256"],
                "config_sha256": spec["config_sha256"],
                "onnx_sha256": None if is_native else spec["onnx_sha256"],
                "engine_sha256": engine_sha,
                "measurement_platform": ["Linux", "aarch64", "NVIDIA Jetson Orin"],
            },
        )

        ap_dir = root / f"05_full2170_ap/{arm}"
        prediction = ap_dir / "prediction_manifest.json"
        ids = canonical_ids()
        write_json(
            prediction,
            {
                "dataset": "OPV2V",
                "split": "test",
                "dataset_samples": 2170,
                "processed_samples": 2170,
                "sample_ids": ids,
                "sample_ids_sha256": id_sequence_sha(ids),
            },
        )
        metrics = ap_dir / "metrics.json"
        write_json(
            metrics,
            {
                "status": "success_full",
                "dataset": "OPV2V",
                "split": "test",
                "dataset_samples": 2170,
                "processed_samples": 2170,
                "failed_samples": 0,
                "fallback_samples": 0,
                "engine_calls": 0 if is_native else 2170,
                "runtime": "native_fp32" if is_native else "trt",
                "requested_runtime": "native" if is_native else "trt",
                "scope": SCOPE,
                "ap30": 0.70 - position * 0.01,
                "ap50": 0.65 - position * 0.01,
                "ap70": 0.60 - position * 0.01,
                "checkpoint_sha256": spec["checkpoint_sha256"],
                "config_sha256": spec["config_sha256"],
                "engine_sha256": engine_sha,
                "prediction_manifest_sha256": digest(prediction),
                "arm": None if is_native else arm,
                "native_identity": (
                    "exact_checkpoint_pytorch_cuda_fp32_unchanged" if is_native else None
                ),
                "numerical_contract": {
                    "dense_scope_engine_execution": True,
                    "silent_fallback_forbidden": True,
                    "fallback_samples": 0,
                },
                "build_identity": (
                    None
                    if is_native
                    else {
                        "manifest_sha256": manifest_sha,
                        "engine_sha256": engine_sha,
                        "checkpoint_sha256": spec["checkpoint_sha256"],
                        "config_sha256": spec["config_sha256"],
                        "onnx_sha256": spec["onnx_sha256"],
                    }
                ),
            },
        )
        write_json(
            ap_dir / "run_manifest.json",
            {
                "status": "success",
                "scope": SCOPE,
                "runtime": "native_fp32" if is_native else "trt",
                "arm": None if is_native else arm,
                "checkpoint_sha256": spec["checkpoint_sha256"],
                "config_sha256": spec["config_sha256"],
                "engine_sha256": engine_sha,
                "prediction_manifest_sha256": digest(prediction),
                "output_json_sha256": digest(metrics),
            },
        )
    return root, manifest_path


def test_finalizer_accepts_trt85_format_datatype_inspector_schema() -> None:
    finalizer_module._validate_inspector(
        {
            "layers": [
                {
                    "Name": "conv",
                    "Inputs": [{"Format/Datatype": "Row major linear FP32"}],
                    "Outputs": [{"Format/Datatype": "Row major linear FP32"}],
                }
            ],
            "unsupported_fields": [],
        }
    )


def read_failure_code(root: Path) -> str:
    evidence = json.loads((root / "06_final/evidence_manifest.json").read_text(encoding="utf-8"))
    return evidence["failure_code"]


def rewrite_manifest_sha_links(root: Path, manifest_path: Path) -> None:
    manifest_sha = digest(manifest_path)
    for arm in ARMS[1:]:
        build_path = root / f"02_engines/{arm}/build.json"
        build = json.loads(build_path.read_text())
        build["manifest_sha256"] = manifest_sha
        write_json(build_path, build)
        gate_path = root / f"01_numeric_gate/{arm}/gate.json"
        gate = json.loads(gate_path.read_text())
        gate["manifest_sha256"] = manifest_sha
        write_json(gate_path, gate)
        metrics_path = root / f"05_full2170_ap/{arm}/metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["build_identity"]["manifest_sha256"] = manifest_sha
        write_json(metrics_path, metrics)
        run_path = root / f"05_full2170_ap/{arm}/run_manifest.json"
        run = json.loads(run_path.read_text())
        run["output_json_sha256"] = digest(metrics_path)
        write_json(run_path, run)


def test_happy_path_writes_only_five_table2_columns_and_grade_a(tmp_path: Path) -> None:
    root, manifest = create_complete_tree(tmp_path)

    result = finalize(manifest, root, root / "06_final")

    assert result["status"] == "complete"
    assert result["evidence_grade"] == "A"
    table = json.loads((root / "06_final/table2_values.json").read_text())
    assert [row["arm"] for row in table["rows"]] == list(ARMS)
    with (root / "06_final/table2_values.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert list(rows[0]) == [
        "arm",
        "display_label",
        "ap70",
        "latency_median_ms",
        "energy_j",
    ]
    assert [row["display_label"] for row in rows] == [DISPLAY[arm] for arm in ARMS]
    evidence = json.loads((root / "06_final/evidence_manifest.json").read_text())
    assert evidence["manifest_sha256"] == digest(manifest)
    assert evidence["artifacts"]
    assert {path.name for path in (root / "06_final").iterdir()} == {
        "table2_values.json",
        "table2_values.csv",
        "evidence_manifest.json",
    }


@pytest.mark.parametrize(
    ("mutation", "failure_code"),
    [
        ("missing_engine", "source_artifacts_not_restored"),
        ("build_blocker", "build_contract_unsupported"),
        ("missing_rail", "energy_evidence_missing"),
        ("sample_1499", "latency_protocol_mismatch"),
        ("ap_fallback", "engine_fallback_detected"),
        ("wrong_scope", "scope_mismatch"),
        ("wrong_identity", "source_sha_mismatch"),
        ("wrong_joint_precision", "joint_cross_precision_control"),
        ("schedule_fp16", "strict_fp32_inspector_failed"),
        ("incomplete_gate_metrics", "real_heldout_numeric_gate_failed"),
        ("gate_over_tolerance", "real_heldout_numeric_gate_failed"),
        ("wrong_platform", "source_sha_mismatch"),
        ("wrong_ap_scope_contract", "engine_fallback_detected"),
        ("different_prediction_ids", "source_sha_mismatch"),
        ("malformed_heldout_item", "real_heldout_numeric_gate_failed"),
        ("wrong_cache_path", "source_sha_mismatch"),
        ("string_gate_metric", "real_heldout_numeric_gate_failed"),
    ],
)
def test_fail_closed_events_do_not_write_table_values(
    tmp_path: Path, mutation: str, failure_code: str
) -> None:
    root, manifest_path = create_complete_tree(tmp_path)
    if mutation == "missing_engine":
        (root / "02_engines/compression_only/model.engine").unlink()
    elif mutation == "build_blocker":
        write_json(
            root / "02_engines/build_contract_blocker.json",
            {"status": "build_contract_unsupported"},
        )
    elif mutation == "missing_rail":
        path = root / "03_latency/original_default/report.json"
        value = json.loads(path.read_text())
        value["rails"].pop("VIN_SYS_5V0")
        write_json(path, value)
    elif mutation == "sample_1499":
        path = root / "03_latency/original_default/report.json"
        value = json.loads(path.read_text())
        value["sample_count"] = 1499
        value["protocol"]["sample_count"] = 1499
        write_json(path, value)
        write_samples(root / "03_latency/original_default/samples.csv", 1499)
    elif mutation == "ap_fallback":
        path = root / "05_full2170_ap/compression_only/metrics.json"
        value = json.loads(path.read_text())
        value["fallback_samples"] = 1
        write_json(path, value)
    elif mutation == "wrong_scope":
        path = root / "03_latency/compression_only/report.json"
        value = json.loads(path.read_text())
        value["scope"] = "wrong_scope"
        write_json(path, value)
    elif mutation == "wrong_identity":
        path = root / "01_numeric_gate/compression_only/gate.json"
        value = json.loads(path.read_text())
        value["engine_sha256"] = "0" * 64
        write_json(path, value)
    elif mutation == "wrong_joint_precision":
        manifest = json.loads(manifest_path.read_text())
        manifest["arms"]["joint_fp16_control"]["q_mode"] = "int8"
        write_json(manifest_path, manifest)
        rewrite_manifest_sha_links(root, manifest_path)
    elif mutation == "schedule_fp16":
        write_json(
            root / "02_engines/schedule_only/inspector.json",
            {"layers": [{"name": "conv", "precision": "FP16"}], "unsupported_fields": []},
        )
    elif mutation == "incomplete_gate_metrics":
        path = root / "01_numeric_gate/compression_only/gate.json"
        value = json.loads(path.read_text())
        value["items"]["item_0"].pop("max_abs")
        write_json(path, value)
    elif mutation == "gate_over_tolerance":
        path = root / "01_numeric_gate/compression_only/gate.json"
        value = json.loads(path.read_text())
        value["items"]["item_0"]["cosine"] = 0.1
        write_json(path, value)
    elif mutation == "wrong_platform":
        path = root / "03_latency/original_default/report.json"
        value = json.loads(path.read_text())
        value["measurement_platform"] = ["Linux", "aarch64", "generic"]
        write_json(path, value)
    elif mutation == "wrong_ap_scope_contract":
        path = root / "05_full2170_ap/compression_only/metrics.json"
        value = json.loads(path.read_text())
        value["numerical_contract"]["dense_scope_engine_execution"] = False
        write_json(path, value)
        run_path = root / "05_full2170_ap/compression_only/run_manifest.json"
        run = json.loads(run_path.read_text())
        run["output_json_sha256"] = digest(path)
        write_json(run_path, run)
    elif mutation == "different_prediction_ids":
        directory = root / "05_full2170_ap/compression_only"
        prediction_path = directory / "prediction_manifest.json"
        prediction = json.loads(prediction_path.read_text())
        prediction["sample_ids"][0] = "different-test-sample"
        prediction["sample_ids_sha256"] = id_sequence_sha(prediction["sample_ids"])
        write_json(prediction_path, prediction)
        metrics_path = directory / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["prediction_manifest_sha256"] = digest(prediction_path)
        write_json(metrics_path, metrics)
        run_path = directory / "run_manifest.json"
        run = json.loads(run_path.read_text())
        run["prediction_manifest_sha256"] = digest(prediction_path)
        run["output_json_sha256"] = digest(metrics_path)
        write_json(run_path, run)
    elif mutation == "malformed_heldout_item":
        path = root / "01_numeric_gate/heldout_manifest.json"
        value = json.loads(path.read_text())
        value["items"][0] = []
        write_json(path, value)
        sidecar_path = root / "01_numeric_gate/heldout_inputs.heldout.json"
        sidecar = json.loads(sidecar_path.read_text())
        sidecar["capture_provenance_sha256"] = digest(path)
        write_json(sidecar_path, sidecar)
    elif mutation == "wrong_cache_path":
        path = root / "02_engines/compression_only/build.json"
        value = json.loads(path.read_text())
        value["timing_cache"] = str(root / "02_engines/other.cache")
        write_json(path, value)
    elif mutation == "string_gate_metric":
        path = root / "01_numeric_gate/compression_only/gate.json"
        value = json.loads(path.read_text())
        value["items"]["item_0"]["cosine"] = "0.999999"
        write_json(path, value)

    with pytest.raises(FinalizationError) as error:
        finalize(manifest_path, root, root / "06_final")

    assert error.value.code == failure_code
    assert read_failure_code(root) == failure_code
    assert not (root / "06_final/table2_values.json").exists()
    assert not (root / "06_final/table2_values.csv").exists()


def test_manifest_requires_exactly_five_arms(tmp_path: Path) -> None:
    root, manifest_path = create_complete_tree(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["arms"].pop("compression_only")
    write_json(manifest_path, manifest)

    with pytest.raises(FinalizationError) as error:
        finalize(manifest_path, root, root / "06_final")

    assert error.value.code == "source_artifacts_not_restored"


def test_cli_failure_is_nonzero_and_prints_explicit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, manifest_path = create_complete_tree(tmp_path)
    write_json(
        root / "02_engines/build_contract_blocker.json",
        {"status": "build_contract_unsupported"},
    )

    exit_code = finalizer_module.main(
        [
            "--manifest",
            str(manifest_path),
            "--result-root",
            str(root),
            "--output-dir",
            str(root / "06_final"),
        ]
    )

    assert exit_code != 0
    assert "build_contract_unsupported" in capsys.readouterr().err


def test_atomic_failure_does_not_replace_existing_table_values(tmp_path: Path) -> None:
    root, manifest_path = create_complete_tree(tmp_path)
    output = root / "06_final"
    finalize(manifest_path, root, output)
    previous = (output / "table2_values.json").read_bytes()
    path = root / "03_latency/original_default/report.json"
    report = json.loads(path.read_text())
    report["rails"].pop("VIN_SYS_5V0")
    write_json(path, report)

    with pytest.raises(FinalizationError):
        finalize(manifest_path, root, output)

    assert (output / "table2_values.json").read_bytes() == previous
    assert json.loads((output / "evidence_manifest.json").read_text())["status"] == "failure"


def test_atomic_output_failure_cannot_leave_a_complete_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest_path = create_complete_tree(tmp_path)
    output = root / "06_final"
    finalize(manifest_path, root, output)
    atomic_write = finalizer_module._atomic_write

    def fail_csv(path: Path, content: bytes) -> None:
        if path.name == "table2_values.csv":
            raise OSError("injected CSV write failure")
        atomic_write(path, content)

    monkeypatch.setattr(finalizer_module, "_atomic_write", fail_csv)
    with pytest.raises(FinalizationError) as error:
        finalize(manifest_path, root, output)

    assert error.value.code == "finalization_io_failed"
    marker = json.loads((output / "evidence_manifest.json").read_text())
    assert marker["status"] == "failure"
    assert marker["evidence_grade"] == "Invalid"
