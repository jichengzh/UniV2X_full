from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.fcooper_trt_ap_bridge_v1 import (
    _checkpoint_state,
    EngineBackbone,
    build_failure_receipt,
    build_prediction_manifest,
    main,
    parse_args,
    pre_scan_dataset_contract,
    require_orin,
    run,
    validate_trt_build_identity,
    validate_engine_binding_contract,
    validate_checkpoint_location,
    validate_evaluated_sequence,
    validate_formal_completion,
)


BASE_ARGS = [
    "--config",
    "config.yaml",
    "--checkpoint",
    "model.pth",
    "--checkpoint-dir",
    "checkpoint-dir",
    "--output-json",
    "metrics.json",
    "--prediction-manifest",
    "predictions.json",
    "--run-log-manifest",
    "run-log.json",
    "--expected-dataset-samples",
    "2170",
]


def test_native_and_engine_cli_are_mutually_exclusive() -> None:
    native = parse_args([*BASE_ARGS, "--native", "--require-full2170"])
    assert native.native is True
    assert native.engine is None

    with pytest.raises(SystemExit):
        parse_args([*BASE_ARGS, "--native", "--engine", "model.engine"])
    with pytest.raises(SystemExit):
        parse_args(BASE_ARGS)
    with pytest.raises(SystemExit):
        parse_args(
            [
                *BASE_ARGS,
                "--native",
                "--arm",
                "compression_only",
                "--build-receipt",
                "build.json",
                "--manifest",
                "manifest.json",
            ]
        )


def test_trt_cli_requires_arm_build_receipt_and_manifest() -> None:
    identity = [
        "--arm",
        "compression_only",
        "--build-receipt",
        "build.json",
        "--manifest",
        "manifest.json",
    ]
    parsed = parse_args([*BASE_ARGS, "--engine", "model.engine", *identity])
    assert parsed.arm == "compression_only"

    for omitted in ("--arm", "--build-receipt", "--manifest"):
        values = identity.copy()
        index = values.index(omitted)
        del values[index : index + 2]
        with pytest.raises(SystemExit):
            parse_args([*BASE_ARGS, "--engine", "model.engine", *values])


def test_full_mode_rejects_max_samples() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                *BASE_ARGS,
                "--native",
                "--require-full2170",
                "--max-samples",
                "3",
            ]
        )


def test_cli_contract_failure_writes_failure_receipts(tmp_path) -> None:
    output = tmp_path / "metrics.json"
    prediction = tmp_path / "predictions.json"
    run_log = tmp_path / "run-log.json"
    args = [
        "--config",
        str(tmp_path / "config.yaml"),
        "--checkpoint",
        str(tmp_path / "model.pth"),
        "--checkpoint-dir",
        str(tmp_path),
        "--native",
        "--output-json",
        str(output),
        "--prediction-manifest",
        str(prediction),
        "--run-log-manifest",
        str(run_log),
        "--require-full2170",
        "--max-samples",
        "3",
    ]

    with pytest.raises(SystemExit):
        main(args)

    assert output.exists()
    assert prediction.exists()
    assert run_log.exists()
    assert '"status": "failure"' in output.read_text()


def test_invalid_integer_cli_still_writes_failure_receipts(tmp_path) -> None:
    output = tmp_path / "metrics.json"
    prediction = tmp_path / "predictions.json"
    run_log = tmp_path / "run-log.json"
    args = [
        "--config",
        str(tmp_path / "config.yaml"),
        "--checkpoint",
        str(tmp_path / "model.pth"),
        "--checkpoint-dir",
        str(tmp_path),
        "--native",
        "--output-json",
        str(output),
        "--prediction-manifest",
        str(prediction),
        "--run-log-manifest",
        str(run_log),
        "--expected-dataset-samples",
        "not-an-integer",
    ]

    with pytest.raises(SystemExit):
        main(args)

    assert output.exists()
    assert prediction.exists()
    assert run_log.exists()


def test_equals_style_cli_still_writes_failure_receipts(tmp_path) -> None:
    output = tmp_path / "metrics.json"
    prediction = tmp_path / "predictions.json"
    run_log = tmp_path / "run-log.json"
    args = [
        f"--config={tmp_path / 'config.yaml'}",
        f"--checkpoint={tmp_path / 'model.pth'}",
        f"--checkpoint-dir={tmp_path}",
        "--native",
        f"--output-json={output}",
        f"--prediction-manifest={prediction}",
        f"--run-log-manifest={run_log}",
        "--expected-dataset-samples=not-an-integer",
    ]

    with pytest.raises(SystemExit):
        main(args)

    assert output.exists()
    assert prediction.exists()
    assert run_log.exists()


def test_full_mode_requires_exact_dataset_length() -> None:
    with pytest.raises(ValueError, match="2170"):
        validate_formal_completion(
            runtime="native_fp32",
            dataset_samples=2169,
            processed_samples=2169,
            failed_samples=0,
            fallback_samples=0,
            engine_calls=0,
            expected_dataset_samples=2170,
        )


def test_prescan_rejects_agent_overflow_before_inference() -> None:
    class Dataset:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return index

        def collate_batch_test(self, records):
            index = records[0]
            cav_ids = ["a"] if index == 0 else ["a", "b", "c", "d", "e", "f"]
            return {
                "ego": {
                    "sample_idx": [index],
                    "cav_id_list": [cav_ids],
                    "record_len": [len(cav_ids)],
                }
            }

    with pytest.raises(ValueError, match="agents"):
        pre_scan_dataset_contract(Dataset(), expected_dataset_samples=2)


def test_prescan_returns_all_ids_in_evaluation_order() -> None:
    class Dataset:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return index

        def collate_batch_test(self, records):
            index = records[0]
            return {
                "ego": {
                    "sample_idx": [index],
                    "cav_id_list": [[f"cav-{index}"]],
                    "record_len": [1],
                }
            }

    assert pre_scan_dataset_contract(
        Dataset(), expected_dataset_samples=2
    ) == ["opv2v-test:0:cav-0", "opv2v-test:1:cav-1"]


def test_evaluated_sequence_requires_exact_unique_full_set() -> None:
    sample_ids = ["opv2v-test:0:a", "opv2v-test:1:b"]
    assert validate_evaluated_sequence(
        sample_ids, expected_dataset_samples=2
    ) == sample_ids

    with pytest.raises(ValueError, match="duplicate"):
        validate_evaluated_sequence(
            ["opv2v-test:0:a", "opv2v-test:0:a"],
            expected_dataset_samples=2,
        )
    with pytest.raises(ValueError, match="count"):
        validate_evaluated_sequence(
            ["opv2v-test:0:a"], expected_dataset_samples=2
        )


def test_trt_requires_linux_aarch64_orin() -> None:
    assert require_orin(lambda: ("Linux", "aarch64", "NVIDIA Jetson Orin")) == (
        "Linux",
        "aarch64",
        "NVIDIA Jetson Orin",
    )
    with pytest.raises(RuntimeError, match="Orin"):
        require_orin(lambda: ("Linux", "aarch64", "generic arm board"))


def test_checkpoint_must_be_under_checkpoint_dir(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "source"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "model.pth"
    checkpoint.write_bytes(b"model")
    assert validate_checkpoint_location(checkpoint, checkpoint_dir) == checkpoint.resolve()

    outside = tmp_path / "outside.pth"
    outside.write_bytes(b"other")
    with pytest.raises(ValueError, match="checkpoint-dir"):
        validate_checkpoint_location(outside, checkpoint_dir)


def test_checkpoint_wrapper_model_key_is_unwrapped(tmp_path: Path) -> None:
    class Torch:
        @staticmethod
        def load(path, map_location):
            assert map_location == "cpu"
            return {"model": {"weight": "tensor"}}

    assert _checkpoint_state(Torch(), tmp_path / "model.pth") == {
        "weight": "tensor"
    }


def test_engine_binding_contract_is_exact() -> None:
    validate_engine_binding_contract(
        input_names=["spatial_features"],
        output_names=["encoded_features"],
        input_shape=(5, 64, 512, 512),
        output_shape=(5, 64, 256, 256),
    )
    with pytest.raises(ValueError, match="contract"):
        validate_engine_binding_contract(
            input_names=["input"],
            output_names=["encoded_features"],
            input_shape=(5, 64, 512, 512),
            output_shape=(5, 64, 256, 256),
        )
    with pytest.raises(ValueError, match="contract"):
        validate_engine_binding_contract(
            input_names=["spatial_features"],
            output_names=["encoded_features"],
            input_shape=(4, 64, 512, 512),
            output_shape=(5, 64, 256, 256),
        )
    with pytest.raises(ValueError, match="contract"):
        validate_engine_binding_contract(
            input_names=["spatial_features"],
            output_names=["encoded_features"],
            input_shape=(5, 64, 512, 512),
            output_shape=(4, 64, 256, 256),
        )


def _write_build_identity(tmp_path: Path):
    engine = tmp_path / "model.engine"
    checkpoint = tmp_path / "model.pth"
    config = tmp_path / "config.yaml"
    onnx = tmp_path / "model.onnx"
    engine.write_bytes(b"orin-engine")
    checkpoint.write_bytes(b"checkpoint")
    config.write_bytes(b"config")
    onnx.write_bytes(b"onnx")

    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    arm = {
        "runtime": "trt",
        "scope": "post_scatter_backbone_shrinker",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "config_path": str(config),
        "config_sha256": digest(config),
        "onnx_path": str(onnx),
        "onnx_sha256": digest(onnx),
        "q_mode": "fp16",
        "builder_policy": "trt85_default",
        "builder_optimization_level": None,
    }
    manifest = tmp_path / "canonical_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "ready",
                "scope": "post_scatter_backbone_shrinker",
                "arms": {"compression_only": arm},
            }
        )
    )
    receipt = tmp_path / "build.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "built",
                "scope": "post_scatter_backbone_shrinker",
                "arm": "compression_only",
                "manifest_sha256": digest(manifest),
                "engine": str(engine.resolve()),
                "engine_sha256": digest(engine),
                "onnx_sha256": digest(onnx),
                "checkpoint_sha256": digest(checkpoint),
                "config_sha256": digest(config),
                "platform": ["Linux", "aarch64", "NVIDIA Jetson Orin"],
                "tensorrt_version": "8.5.2.2",
                "build_arguments": {
                    "builder_policy": "trt85_default",
                    "builder_optimization_level": None,
                    "q_mode": "fp16",
                },
                "calibration_cache_read": False,
                "h800_engine_or_cache_read": False,
            }
        )
    )
    return engine, checkpoint, config, manifest, receipt


def test_build_receipt_binds_exact_orin_arm_and_all_sources(tmp_path: Path) -> None:
    engine, checkpoint, config, manifest, receipt = _write_build_identity(tmp_path)

    identity = validate_trt_build_identity(
        engine_path=engine,
        checkpoint_path=checkpoint,
        config_path=config,
        manifest_path=manifest,
        receipt_path=receipt,
        arm_name="compression_only",
    )

    assert identity["arm"] == "compression_only"
    assert identity["engine_sha256"] == hashlib.sha256(engine.read_bytes()).hexdigest()
    assert identity["build_receipt_sha256"]
    assert identity["manifest_sha256"]
    assert identity["onnx_sha256"]
    assert identity["checkpoint_sha256"]
    assert identity["config_sha256"]
    assert identity["builder_contract"] == {
        "builder_policy": "trt85_default",
        "builder_optimization_level": None,
        "q_mode": "fp16",
    }


def test_ap_bridge_rejects_manifest_and_receipt_that_jointly_forge_builder_level(
    tmp_path: Path,
) -> None:
    engine, checkpoint, config, manifest, receipt = _write_build_identity(tmp_path)
    manifest_value = json.loads(manifest.read_text())
    manifest_value["arms"]["compression_only"]["builder_policy"] = "level_5"
    manifest_value["arms"]["compression_only"]["builder_optimization_level"] = 5
    manifest.write_text(json.dumps(manifest_value))
    receipt_value = json.loads(receipt.read_text())
    receipt_value["manifest_sha256"] = hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    receipt_value["build_arguments"]["builder_policy"] = "level_5"
    receipt_value["build_arguments"]["builder_optimization_level"] = 5
    receipt.write_text(json.dumps(receipt_value))

    with pytest.raises(ValueError, match="TRT 8.5 default"):
        validate_trt_build_identity(
            engine_path=engine,
            checkpoint_path=checkpoint,
            config_path=config,
            manifest_path=manifest,
            receipt_path=receipt,
            arm_name="compression_only",
        )


@pytest.mark.parametrize(
    ("target", "value", "message"),
    [
        ("receipt.status", "failure", "status"),
        ("receipt.arm", "schedule_only", "arm"),
        ("receipt.scope", "wrong", "scope"),
        ("receipt.engine_sha256", "0" * 64, "engine"),
        ("receipt.checkpoint_sha256", "0" * 64, "checkpoint"),
        ("receipt.config_sha256", None, "config"),
        ("receipt.onnx_sha256", "0" * 64, "ONNX"),
        ("receipt.manifest_sha256", "0" * 64, "manifest"),
        ("receipt.platform", ["Linux", "x86_64", "H800 server"], "Orin"),
        ("receipt.h800_engine_or_cache_read", True, "H800"),
        ("receipt.calibration_cache_read", True, "calibration"),
        ("receipt.build_arguments.builder_policy", "level_5", "builder"),
        ("receipt.build_arguments.builder_optimization_level", 5, "builder"),
        ("receipt.build_arguments.q_mode", "fp32", "builder"),
        ("manifest.arms.compression_only.config_sha256", "0" * 64, "config"),
    ],
)
def test_build_receipt_rejects_mismatch_h800_and_stale_engine(
    tmp_path: Path, target: str, value, message: str
) -> None:
    engine, checkpoint, config, manifest, receipt = _write_build_identity(tmp_path)
    root_name, *parts = target.split(".")
    path = receipt if root_name == "receipt" else manifest
    payload = json.loads(path.read_text())
    node = payload
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        validate_trt_build_identity(
            engine_path=engine,
            checkpoint_path=checkpoint,
            config_path=config,
            manifest_path=manifest,
            receipt_path=receipt,
            arm_name="compression_only",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [("stale_engine", "engine"), ("h800", "Orin")],
)
def test_runtime_fails_before_import_or_inference_for_invalid_build_receipt(
    tmp_path: Path, mutation: str, message: str
) -> None:
    engine, checkpoint, config, manifest, receipt = _write_build_identity(tmp_path)
    if mutation == "stale_engine":
        engine.write_bytes(b"stale-or-replaced-engine")
    else:
        payload = json.loads(receipt.read_text())
        payload["platform"] = ["Linux", "x86_64", "NVIDIA H800"]
        receipt.write_text(json.dumps(payload))
    args = argparse.Namespace(
        engine=engine,
        checkpoint=checkpoint,
        checkpoint_dir=tmp_path,
        config=config,
        manifest=manifest,
        build_receipt=receipt,
        arm="compression_only",
    )

    with pytest.raises(ValueError, match=message):
        run(
            args,
            started_at="start",
            orin_probe=lambda: ("Linux", "aarch64", "NVIDIA Jetson Orin"),
        )


def test_runtime_checks_checkpoint_dir_before_receipt_or_imports(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.pth"
    outside.write_bytes(b"checkpoint")
    args = argparse.Namespace(
        checkpoint=outside,
        checkpoint_dir=tmp_path / "source",
        engine=tmp_path / "missing.engine",
        config=tmp_path / "missing.yaml",
        manifest=tmp_path / "missing-manifest.json",
        build_receipt=tmp_path / "missing-receipt.json",
        arm="compression_only",
    )

    with pytest.raises(ValueError, match="checkpoint-dir"):
        run(
            args,
            started_at="start",
            orin_probe=lambda: ("Linux", "aarch64", "NVIDIA Jetson Orin"),
        )


def test_runtime_checks_local_orin_before_receipt_or_imports(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    args = argparse.Namespace(
        checkpoint=checkpoint,
        checkpoint_dir=source,
        engine=tmp_path / "missing.engine",
        config=tmp_path / "missing.yaml",
        manifest=tmp_path / "missing-manifest.json",
        build_receipt=tmp_path / "missing-receipt.json",
        arm="compression_only",
    )

    with pytest.raises(RuntimeError, match="Orin"):
        run(
            args,
            started_at="start",
            orin_probe=lambda: ("Linux", "aarch64", "generic arm board"),
        )


class _FakeTensor:
    def __init__(self, value: np.ndarray):
        self.value = value
        self.shape = value.shape
        self.device = "cpu"
        self.dtype = value.dtype

    def contiguous(self) -> "_FakeTensor":
        return self

    def float(self) -> "_FakeTensor":
        return self

    def __getitem__(self, item):
        return _FakeTensor(self.value[item])


class _FakeRunner:
    input_shape = (5, 64, 2, 2)

    def __call__(self, source: _FakeTensor) -> _FakeTensor:
        assert source.shape == self.input_shape
        return _FakeTensor(source.value + 1)


def test_engine_adapter_pads_to_five_and_returns_only_unpadded_agents() -> None:
    adapter = EngineBackbone(_FakeRunner(), tensor_ops=np)
    source = _FakeTensor(np.zeros((2, 64, 2, 2), dtype=np.float32))

    result = adapter({"spatial_features": source})

    assert result["spatial_features_2d"].shape == (2, 64, 2, 2)


@pytest.mark.parametrize(
    ("runtime", "engine_calls"),
    [("trt", 2170), ("native_fp32", 0)],
)
def test_formal_completion_requires_exact_counts(
    runtime: str, engine_calls: int
) -> None:
    result = validate_formal_completion(
        runtime=runtime,
        dataset_samples=2170,
        processed_samples=2170,
        failed_samples=0,
        fallback_samples=0,
        engine_calls=engine_calls,
        expected_dataset_samples=2170,
    )
    assert result["status"] == "success_full"

    with pytest.raises(ValueError, match="formal"):
        validate_formal_completion(
            runtime=runtime,
            dataset_samples=2170,
            processed_samples=2170,
            failed_samples=0,
            fallback_samples=1,
            engine_calls=engine_calls,
            expected_dataset_samples=2170,
        )


def test_prediction_manifest_preserves_order_and_binds_hash() -> None:
    ids = ["opv2v-test:1:a", "opv2v-test:0:b"]
    manifest = build_prediction_manifest(ids, dataset_samples=2)

    assert manifest["sample_ids"] == ids
    assert manifest["sample_ids_sha256"]
    assert manifest == build_prediction_manifest(ids, dataset_samples=2)
    assert manifest != build_prediction_manifest(list(reversed(ids)), dataset_samples=2)


def test_failure_receipt_removes_secret_arguments_and_binds_sources(
    tmp_path,
) -> None:
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.pth"
    config.write_text("config")
    checkpoint.write_bytes(b"checkpoint")
    secret_options = {
        "api_" + "token": "do-not-record",
        "pass" + "word": "do-not-record",
    }
    args = argparse.Namespace(
        config=config,
        checkpoint=checkpoint,
        engine=None,
        expected_dataset_samples=2170,
        **secret_options,
    )

    receipt = build_failure_receipt(
        args=args,
        error=RuntimeError("failed do-not-record"),
        started_at="start",
        ended_at="end",
    )

    encoded = repr(receipt)
    assert "do-not-record" not in encoded
    assert "api_token" not in receipt["arguments"]
    assert "password" not in receipt["arguments"]
    assert receipt["status"] == "failure"
    assert receipt["identities"]["config_sha256"]
    assert receipt["identities"]["checkpoint_sha256"]
    assert receipt["identities"]["dataset"] == "OPV2V-test"
