import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


MODULE = Path(__file__).parents[2] / "tools/orin_deploy/fcooper_orin_five_config.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fcooper_manifest", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_source_tree(root, module):
    global_files = {
        "contracts/formal_contract_v2.json": b"formal",
        "contracts/frozen_contract.json": b"config",
        "contracts/opv2v_test_manifest_fresh.txt": b"held-out",
        "scanner/fcooper_partition.yaml": b"partition",
        "contracts/recovery_training_contract.json": b"recovery",
        "final/fcooper_stage6_five_arm_audit_v2.json": b"audit",
        "final/fcooper_stage6_trt_delta_ap_0.10_v2.csv": b"csv",
    }
    identities = {}
    for relative, contents in global_files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        identities[relative] = digest(path)
    module.GLOBAL_IDENTITIES = identities
    patched_arms = {}
    for index, (name, arm) in enumerate(module.FROZEN_ARMS.items()):
        source = root / "search_artifacts/sources" / "x".join(map(str, arm["widths"]))
        source.mkdir(parents=True, exist_ok=True)
        config = source / "config.yaml"
        checkpoint = source / "recovered_checkpoint.pth"
        onnx = source / f"fcooper_dense_{index}.onnx"
        if not config.exists():
            config.write_bytes(f"config-{name}".encode())
        if not checkpoint.exists():
            checkpoint.write_bytes(f"checkpoint-{name}".encode())
        if name != "original_default":
            onnx.write_bytes(f"onnx-{name}".encode())
        patched = dict(arm)
        patched["checkpoint_sha256"] = digest(checkpoint)
        if "onnx_sha256" in arm:
            patched["onnx_sha256"] = digest(onnx)
        patched_arms[name] = patched
        report = {
            "status": "success",
            "formal_measurement_eligible": True,
            "widths": arm["widths"],
            "input_name": "spatial_features",
            "input_shape": [5, 64, 512, 512],
            "checkpoint_sha256": digest(checkpoint),
            "config_sha256": digest(config),
            "outputs": [{"name": "encoded_features", "shape": [5, 64, 32, 32]}],
        }
        if onnx.exists():
            report["onnx_sha256"] = digest(onnx)
        (source / "source_export_report.json").write_text(json.dumps(report), encoding="utf-8")
    module.FROZEN_ARMS = patched_arms
    original = root / "search_artifacts/sources/64x128x256x128x256/config.yaml"
    module.ORIGINAL_CONFIG_SHA256 = digest(original)
    module._onnx_output_contract = lambda _path: {"name": "encoded_features", "shape": [5, 64, 32, 32], "dtype": "float32"}


def test_manifest_success_has_exactly_five_frozen_arms_and_deterministic_sums(tmp_path):
    module = load_module()
    source = tmp_path / "source"
    write_source_tree(source, module)
    output, sums = tmp_path / "manifest.json", tmp_path / "sha256sum.txt"

    manifest = module.create_manifest(source, output, sums)

    assert manifest["status"] == "ready"
    assert set(manifest["arms"]) == set(module.FROZEN_ARMS)
    assert manifest["input_shape"] == [5, 64, 512, 512]
    assert "sha256sum.txt" not in sums.read_text(encoding="utf-8")
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "ready"


@pytest.mark.parametrize("mutation, expected", [("missing", "source_artifacts_not_restored"), ("hash", "source_sha_mismatch")])
def test_manifest_writes_fail_closed_receipts(tmp_path, mutation, expected):
    module = load_module()
    source = tmp_path / "source"
    write_source_tree(source, module)
    if mutation == "missing":
        (source / "contracts/formal_contract_v2.json").unlink()
    else:
        (source / "contracts/formal_contract_v2.json").write_text("drift", encoding="utf-8")
    output = tmp_path / "receipt.json"

    with pytest.raises(module.ManifestError) as error:
        module.create_manifest(source, output, tmp_path / "sums.txt")

    assert error.value.status == expected
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == expected


@pytest.mark.parametrize("forbidden", ["foreign.engine", "timing.cache", "calibration_cache.bin", "H800_engine.plan"])
def test_manifest_rejects_forbidden_engine_and_cache_artifacts(tmp_path, forbidden):
    module = load_module()
    source = tmp_path / "source"
    write_source_tree(source, module)
    (source / forbidden).write_bytes(b"not allowed")

    with pytest.raises(module.ManifestError) as error:
        module.create_manifest(source, tmp_path / "receipt.json", tmp_path / "sums.txt")

    assert error.value.status == "source_sha_mismatch"


def test_frozen_arm_semantics_cannot_be_overridden():
    module = load_module()
    assert module.FROZEN_ARMS["original_default"]["runtime"] == "native"
    for name in (
        "compression_only",
        "schedule_only",
        "compress_then_tune",
        "joint_fp16_control",
    ):
        arm = module.FROZEN_ARMS[name]
        assert arm["builder_policy"] == "trt85_default"
        assert arm["builder_optimization_level"] is None
    assert module.FROZEN_ARMS["schedule_only"]["strict_fp32"] is True
    assert module.FROZEN_ARMS["joint_fp16_control"]["joint_cross_precision_control"] is True


def test_manifest_accepts_real_legacy_export_report_schema_and_binds_onnx_output(tmp_path, monkeypatch):
    module = load_module()
    source = tmp_path / "source"
    write_source_tree(source, module)
    report = source / "search_artifacts/sources/32x64x64x32x64/source_export_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload.pop("outputs")
    payload["width"] = payload.pop("widths")
    payload["output_shape"] = [5, 64, 32, 32]
    report.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(module, "_onnx_output_contract", lambda _path: {"name": "encoded_features", "shape": [5, 64, 32, 32], "dtype": "float32"})

    manifest = module.create_manifest(source, tmp_path / "manifest.json", tmp_path / "sums.txt")

    arm = manifest["arms"]["compression_only"]
    assert arm["output_name"] == "encoded_features"
    assert arm["output_dtype"] == "float32"


def test_present_wrong_onnx_sha_is_mismatch_not_not_restored(tmp_path):
    module = load_module()
    source = tmp_path / "source"
    write_source_tree(source, module)
    module.FROZEN_ARMS["compression_only"]["onnx_sha256"] = "0" * 64

    with pytest.raises(module.ManifestError) as error:
        module.create_manifest(source, tmp_path / "receipt.json", tmp_path / "sums.txt")

    assert error.value.status == "source_sha_mismatch"


def test_manifest_records_symbolic_raw_onnx_output_and_rejects_ambiguous_native_onnx(tmp_path, monkeypatch):
    module = load_module()
    source = tmp_path / "source"
    write_source_tree(source, module)
    monkeypatch.setattr(module, "_onnx_output_contract", lambda _path: {"name": "encoded_features", "shape": [None, 64, 32, 32], "dtype": "float32"})
    manifest = module.create_manifest(source, tmp_path / "manifest.json", tmp_path / "sums.txt")
    arm = manifest["arms"]["compression_only"]
    assert arm["raw_onnx_output_shape"] == [None, 64, 32, 32]
    assert arm["output_contract_source"] == "source_export_report"
    assert arm["requires_trt_static_resolution"] is True
    module = load_module()
    tree = tmp_path / "ambiguous"
    tree.mkdir()
    (tree / "fcooper_dense_a.onnx").write_bytes(b"a")
    (tree / "fcooper_dense_b.onnx").write_bytes(b"b")
    with pytest.raises(module.ManifestError, match="exactly one"):
        module._find_one(tree, "fcooper_dense_*.onnx", "ONNX")
