import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


MODULE = Path(__file__).parents[2] / "tools/orin_deploy/fcooper_orin_runner.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fcooper_runner", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_preflight_refuses_native_arm_wrong_platform_existing_outputs_and_contract(tmp_path):
    module = load_module()
    manifest = {"arms": {
        "original_default": {"runtime": "native"},
        "compression_only": {"runtime": "trt", "builder_policy": "trt85_default",
                              "builder_optimization_level": None,
                              "input_name": "spatial_features", "input_shape": [5, 64, 512, 512],
                              "output_name": "encoded_features"},
    }}
    args = {"engine": tmp_path / "engine.plan", "timing_cache": tmp_path / "cache.bin"}
    with pytest.raises(module.RunnerError, match="original_default"):
        module.validate_build_preflight(manifest, "original_default", args, lambda: ("Linux", "aarch64", "Orin"), {"inputs": [], "outputs": []})
    with pytest.raises(module.RunnerError, match="aarch64/Orin"):
        module.validate_build_preflight(manifest, "compression_only", args, lambda: ("Linux", "x86_64", "server"), {"inputs": [], "outputs": []})
    args["engine"].write_bytes(b"old")
    with pytest.raises(module.RunnerError, match="pre-existing"):
        module.validate_build_preflight(manifest, "compression_only", args, lambda: ("Linux", "aarch64", "Orin"), {"inputs": [], "outputs": []})
    args["engine"].unlink()
    with pytest.raises(module.RunnerError, match="exactly one"):
        module.validate_build_preflight(manifest, "compression_only", args, lambda: ("Linux", "aarch64", "Orin"), {"inputs": [], "outputs": []})


def test_builder_policy_matches_pyramid_trt85_default_and_rejects_fabricated_levels():
    module = load_module()
    module.validate_builder_policy(
        {
            "runtime": "trt",
            "builder_policy": "trt85_default",
            "builder_optimization_level": None,
        }
    )
    for arm in (
        {
            "runtime": "trt",
            "builder_policy": "trt85_default",
            "builder_optimization_level": 5,
        },
        {
            "runtime": "trt",
            "builder_policy": "level_0",
            "builder_optimization_level": None,
        },
    ):
        with pytest.raises(module.RunnerError, match="TRT 8.5 default builder"):
            module.validate_builder_policy(arm)


def test_artifact_root_containment_does_not_depend_on_path_is_relative_to(tmp_path):
    module = load_module()
    root, inside, outside = tmp_path / "root", tmp_path / "root/out.bin", tmp_path / "elsewhere/out.bin"
    assert module.path_is_within(inside, root)
    assert not module.path_is_within(outside, root)


def test_schedule_only_inspector_fails_on_reduced_precision():
    module = load_module()
    with pytest.raises(module.RunnerError, match="strict FP32"):
        module.validate_schedule_inspector({"layers": [{"name": "conv", "precision": "FP16"}]})
    with pytest.raises(module.RunnerError, match="strict FP32"):
        module.validate_schedule_inspector({"layers": [{"name": "conv"}]})
    with pytest.raises(module.RunnerError, match="strict FP32"):
        module.validate_schedule_inspector({"layers": []})
    module.validate_schedule_inspector({"layers": [{"name": "conv", "precision": "FP32"}]})
    module.validate_schedule_inspector({"layers": [{"name": "fp16_named_but_fp32", "precision": "FP32", "compute_precision": "FP32"}]})


def test_schedule_inspector_accepts_trt85_format_datatype_schema():
    module = load_module()
    fp32_layer = {
        "Name": "conv",
        "Inputs": [{"Format/Datatype": "Row major linear FP32"}],
        "Outputs": [{"Format/Datatype": "Row major linear FP32"}],
    }
    module.validate_schedule_inspector({"layers": [fp32_layer]})
    fp16_layer = {
        **fp32_layer,
        "Outputs": [{"Format/Datatype": "Row major linear FP16"}],
    }
    with pytest.raises(module.RunnerError, match="reduced precision"):
        module.validate_schedule_inspector({"layers": [fp16_layer]})


def test_build_preflight_requires_static_float_output_contract(tmp_path):
    module = load_module()
    manifest = {"arms": {"compression_only": {"runtime": "trt", "builder_policy": "trt85_default", "builder_optimization_level": None, "input_name": "spatial_features", "input_shape": [5, 64, 512, 512], "output_name": "encoded_features", "output_shape": [5, 64, 32, 32], "output_dtype": "float32"}}}
    paths = {"engine": tmp_path / "e", "timing_cache": tmp_path / "c"}
    contract = {"inputs": [{"name": "spatial_features", "shape": [5, 64, 512, 512], "dtype": "float32"}], "outputs": [{"name": "encoded_features", "shape": [5, 64, None, 32], "dtype": "float16"}]}
    with pytest.raises(module.RunnerError, match="output contract"):
        module.validate_build_preflight(manifest, "compression_only", paths, lambda: ("Linux", "aarch64", "Orin"), contract)
    contract["outputs"][0] = {"name": "encoded_features", "shape": [5, 64, None, 32], "dtype": "float32"}
    module.validate_build_preflight(manifest, "compression_only", paths, lambda: ("Linux", "aarch64", "Orin"), contract)
    class Tensor:
        def __init__(self, name, shape, dtype): self.name, self.shape, self.dtype = name, shape, dtype
    class Network:
        num_inputs, num_outputs = 1, 1
        def get_input(self, _): return Tensor("spatial_features", (5, 64, 512, 512), "float32")
        def get_output(self, _): return Tensor("encoded_features", (5, 64, None, 32), "float32")
    with pytest.raises(module.RunnerError, match="output contract"):
        module.validate_trt_network_contract(manifest["arms"]["compression_only"], Network())


def test_trt_float_dtype_accepts_tensorrt_85_datatype_float_repr():
    module = load_module()
    class Dtype:
        def __str__(self): return "DataType.FLOAT"
    assert module.is_trt_float32(Dtype())


def test_native_scope_uses_fcooper_dict_semantics():
    module = load_module()
    seen = {}
    class Backbone:
        def __call__(self, value):
            seen["backbone"] = value
            return {"spatial_features_2d": "encoded"}
    class Shrinker:
        def __call__(self, value):
            seen["shrinker"] = value
            return "result"
    model = type("Model", (), {"backbone_m1": Backbone(), "shrinker_m1": Shrinker()})()
    assert module._run_native_scope(model, "tensor") == "result"
    assert seen == {"backbone": {"spatial_features": "tensor"}, "shrinker": "encoded"}


def test_build_receipt_identity_and_atomic_failure_cleanup(tmp_path):
    module = load_module()
    engine, cache, inspector, receipt = (tmp_path / name for name in ("engine", "cache", "inspector.json", "receipt.json"))
    with pytest.raises(module.RunnerError):
        module.commit_build_artifacts({"engine": engine, "timing_cache": cache, "inspector_json": inspector, "artifact_json": receipt}, b"engine", b"cache", {"layers": [{"precision": "FP16"}]}, {"arm": "schedule_only"}, strict_fp32=True)
    assert not any(path.exists() for path in (engine, cache, inspector, receipt))
    arm = {
        "q_mode": "fp16",
        "builder_policy": "trt85_default",
        "builder_optimization_level": None,
        "onnx_sha256": "o",
        "config_sha256": "c",
        "checkpoint_sha256": "p",
    }
    receipt_value = {
        "status": "built",
        "arm": "compression_only",
        "manifest_sha256": "m",
        "onnx_sha256": "o",
        "config_sha256": "c",
        "checkpoint_sha256": "p",
        "engine_sha256": "e",
        "tensorrt_version": "8.5.2.2",
        "platform": ["Linux", "aarch64", "NVIDIA Jetson Orin"],
        "build_arguments": {
            "builder_policy": "trt85_default",
            "builder_optimization_level": None,
            "q_mode": "fp16",
        },
        "h800_engine_or_cache_read": False,
        "calibration_cache_read": False,
    }
    built = module.validate_build_receipt(
        receipt_value, "compression_only", "m", arm, "e"
    )
    assert built["arm"] == "compression_only"
    with pytest.raises(module.RunnerError, match="receipt identity"):
        module.validate_build_receipt(
            {**receipt_value, "config_sha256": "wrong"},
            "compression_only",
            "m",
            arm,
            "e",
        )
    for mutation in (
        {"tensorrt_version": "10.13.0"},
        {"platform": ["Linux", "x86_64", "server"]},
        {"build_arguments": {"builder_policy": "level_5"}},
        {"h800_engine_or_cache_read": True},
        {"calibration_cache_read": True},
    ):
        with pytest.raises(module.RunnerError):
            module.validate_build_receipt(
                {**receipt_value, **mutation},
                "compression_only",
                "m",
                arm,
                "e",
            )
    assert module.timing_cache_attached(None) is True


def test_heldout_provenance_binds_canonical_opv2v_and_measurement_identity(tmp_path):
    module = load_module()
    inputs = tmp_path / "heldout.npy"
    np.save(inputs, np.zeros((1, 5, 64, 512, 512), dtype=np.float32))
    sidecar = {"input_sha256": module.sha256_file(inputs), "opv2v_test_manifest_sha256": "correct", "capture_provenance_sha256": "capture"}
    module.write_json(inputs.with_suffix(".heldout.json"), sidecar)
    manifest = {"global_identities": {"contracts/opv2v_test_manifest_fresh.txt": {"sha256": "correct"}}}
    assert module.validate_heldout_provenance(manifest, inputs)["capture_provenance_sha256"] == "capture"
    sidecar["opv2v_test_manifest_sha256"] = "wrong"
    module.write_json(inputs.with_suffix(".heldout.json"), sidecar)
    with pytest.raises(module.RunnerError, match="OPV2V"):
        module.validate_heldout_provenance(manifest, inputs)
    sidecar["opv2v_test_manifest_sha256"] = "correct"
    module.write_json(inputs.with_suffix(".heldout.json"), sidecar)
    identity = module.measurement_identity(manifest, inputs)
    assert identity["opv2v_test_manifest_sha256"] == "correct"


def test_numeric_gate_metrics_and_fail_closed_shape_and_nonfinite(tmp_path):
    module = load_module()
    reference = tmp_path / "reference.npz"
    actual = tmp_path / "actual.npz"
    np.savez(reference, item_0=np.array([1.0, 2.0], dtype=np.float32))
    np.savez(actual, item_0=np.array([1.0, 2.01], dtype=np.float32))
    identity = {"manifest_sha256": "manifest", "input_sha256": "input", "opv2v_test_manifest_sha256": "opv2v", "capture_provenance_sha256": "capture", "checkpoint_sha256": "checkpoint", "config_sha256": "config", "onnx_sha256": "onnx"}
    module._write_identity(reference, identity)
    module._write_identity(actual, {**identity, "engine_sha256": "engine"})
    report = module.numeric_gate(reference, actual, tmp_path / "gate.json", {"q_mode": "fp16", "checkpoint_sha256": "checkpoint", "onnx_sha256": "onnx"})
    assert report["status"] == "pass"
    assert report["items"]["item_0"]["MAE"] == pytest.approx(0.005, abs=1e-5)
    np.savez(actual, item_0=np.array([1.0], dtype=np.float32))
    with pytest.raises(module.RunnerError, match="shape"):
        module.numeric_gate(reference, actual, tmp_path / "gate.json", {"q_mode": "fp16", "checkpoint_sha256": "checkpoint", "onnx_sha256": "onnx"})
    np.savez(actual, item_0=np.array([1.0, np.nan], dtype=np.float32))
    with pytest.raises(module.RunnerError, match="nonfinite"):
        module.numeric_gate(reference, actual, tmp_path / "gate.json", {"q_mode": "fp16", "checkpoint_sha256": "checkpoint", "onnx_sha256": "onnx"})


def test_fp32_numeric_gate_matches_existing_pyramid_orin_envelope():
    module = load_module()
    assert module.TOLERANCES["fp32"] == {
        "cosine_min": 0.99999,
        "nrmse_max": 0.01,
        "mae_max": 0.001,
        "max_abs_max": 0.05,
    }


@pytest.mark.parametrize("warmup,iters,repeat", [(0, 300, 5), (20, 1, 5), (20, 300, 1)])
def test_measurement_protocol_is_immutable(warmup, iters, repeat):
    module = load_module()
    with pytest.raises(module.RunnerError):
        module.validate_measurement_protocol(warmup, iters, repeat)
    with pytest.raises(module.RunnerError):
        module.require_sample_count(1499)
    module.validate_measurement_protocol(20, 300, 5)
    module.require_sample_count(1500)


def test_main_rail_energy_formula_fails_closed_and_scrubs_password(tmp_path):
    module = load_module()
    log = tmp_path / "tegrastats.log"
    log.write_text("12:00:00 VIN_SYS_5V0 5000mW VDD_GPU_SOC 2000mW\n12:00:01 VIN_SYS_5V0 7000mW\n", encoding="utf-8")
    rails = module.parse_tegrastats(log)
    assert rails["VIN_SYS_5V0"]["watts"] == [5.0, 7.0]
    assert module.compute_energy_j(rails, 10.0) == pytest.approx(0.06)
    with pytest.raises(module.RunnerError, match="VIN_SYS_5V0"):
        module.compute_energy_j({"VDD_GPU_SOC": {"watts": [2.0], "timestamps": ["x"]}}, 10)
    command, child_env = module.secure_tegrastats_command({"ORIN_SUDO_PW": "secret", "KEEP": "yes"})
    assert "secret" not in json.dumps({"command": command, "env": child_env})
    assert "ORIN_SUDO_PW" not in child_env
    assert command == [
        "sudo",
        "-n",
        "/usr/bin/tegrastats",
        "--interval",
        "500",
    ]
    assert module.tegrastats_stop_command() == [
        "sudo",
        "-n",
        "/usr/bin/tegrastats",
        "--stop",
    ]


def test_measurement_receipt_binds_exact_samples_and_fails_closed_without_main_rail(tmp_path):
    module = load_module()
    output = tmp_path / "measurement.json"
    receipt = module.measurement_receipt(
        output, [1.0] * 1500, {"VIN_SYS_5V0": {"watts": [5.0], "timestamps": ["t"]}},
        {"arm": "compression_only", "scope": "post_scatter_backbone_shrinker"}, "power.log",
    )
    assert receipt["status"] == "complete"
    assert receipt["sample_count"] == 1500
    with pytest.raises(module.RunnerError, match="VIN_SYS_5V0"):
        module.measurement_receipt(output, [1.0] * 1500, {}, {"arm": "compression_only"}, "power.log")
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "energy_evidence_missing"


def test_tegrastats_uses_native_stop_and_never_forwards_password(monkeypatch, tmp_path):
    module = load_module()
    class Process:
        def __init__(self):
            self.stdin = None
            self.wait_calls = []
        def poll(self): return None
        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            return 0
    process = Process()
    popen_calls = []
    run_calls = []
    monkeypatch.setenv("ORIN_SUDO_PW", "secret")
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *args, **kwargs: popen_calls.append((args, kwargs)) or process,
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: run_calls.append((args, kwargs)),
    )
    started, _ = module._start_tegrastats(tmp_path / "power.log")
    assert started is process
    assert "ORIN_SUDO_PW" not in module.os.environ
    module._stop_tegrastats(process, None)
    assert popen_calls[0][0][0] == module.secure_tegrastats_command()[0]
    assert popen_calls[0][1]["stdin"] is module.subprocess.DEVNULL
    assert run_calls[-1][0][0] == module.tegrastats_stop_command()
    assert process.wait_calls == [10]


def test_tegrastats_ready_gate_requires_live_main_rail_before_samples(tmp_path):
    module = load_module()
    class Process:
        def __init__(self, status=None): self.status = status
        def poll(self): return self.status
    ready = tmp_path / "ready.log"
    ready.write_text("12:00:00 VIN_SYS_5V0 5000mW\n", encoding="utf-8")
    module.wait_for_tegrastats_main_rail(Process(), ready, timeout_seconds=0, sleep=lambda _: None)
    missing = tmp_path / "missing.log"
    missing.write_text("12:00:00 VDD_GPU_SOC 5000mW\n", encoding="utf-8")
    with pytest.raises(module.RunnerError, match="VIN_SYS_5V0"):
        module.wait_for_tegrastats_main_rail(Process(), missing, timeout_seconds=0, sleep=lambda _: None)
    with pytest.raises(module.RunnerError, match="exited"):
        module.wait_for_tegrastats_main_rail(Process(1), missing, timeout_seconds=0, sleep=lambda _: None)
