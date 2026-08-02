#!/usr/bin/env python3
"""Create the immutable, Orin-only F-Cooper five-arm manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


INPUT_NAME = "spatial_features"
OUTPUT_NAME = "encoded_features"
INPUT_SHAPE = [5, 64, 512, 512]
SCOPE = "post_scatter_backbone_shrinker"

GLOBAL_IDENTITIES = {
    "contracts/opv2v_test_manifest_fresh.txt": "e70afc9c82d29d405d86aa75fc6651a0fec77e88b825934c6464650f3416aafb",
    "scanner/fcooper_partition.yaml": "5ab65f100fc15338972547b1562d7b9c5c2d3f4a3625ee3409491a76be255364",
    "contracts/recovery_training_contract.json": "4e854d6fa1f0d538301fdf0640e47cf01db09aa0614093d889de05a4dfeae109",
    "final/fcooper_stage6_five_arm_audit_v2.json": "ed5ba5815496215a431930508436e0eec8aada1e12237d6c0964aee0bbbaae61",
    "final/fcooper_stage6_trt_delta_ap_0.10_v2.csv": "5fb5c816fe5f7987afb8c8c5c158c3ccf654d9bd68c2ded6a4b6cf8d64ff7476",
}
ORIGINAL_CONFIG_SHA256 = "8ac5183db729072a4efd50f7eb8b10003263396c6cd7e15d92d520f41cadc1c7"
REQUIRED_CONTRACT_FILES = ("contracts/formal_contract_v2.json", "contracts/frozen_contract.json")

FROZEN_ARMS: dict[str, dict[str, Any]] = {
    "original_default": {"widths": [64, 128, 256, 128, 256], "q_mode": "fp32", "runtime": "native", "checkpoint_sha256": "9ba4786726c71b226ee9ee9561ea82bfe92a21909ddad7653896dd34fb06d0c0"},
    "compression_only": {"widths": [32, 64, 64, 32, 64], "q_mode": "fp16", "runtime": "trt", "builder_policy": "trt85_default", "builder_optimization_level": None, "checkpoint_sha256": "b46497fce1e9529c2748e1777fb1f3c786c41c3e0cacc168ed59f2b4f3a837af", "onnx_sha256": "e91a270037092c342fd7ec36a522dbceea4f4e94f200ec15efd9e4137c14e5eb"},
    "schedule_only": {"widths": [64, 128, 256, 128, 256], "q_mode": "fp32", "runtime": "trt", "builder_policy": "trt85_default", "builder_optimization_level": None, "strict_fp32": True, "checkpoint_sha256": "9ba4786726c71b226ee9ee9561ea82bfe92a21909ddad7653896dd34fb06d0c0", "onnx_sha256": "4370cb078e53e84838414c2c3947b5986bded82f2b217631ad4e8356846840cc"},
    "compress_then_tune": {"widths": [64, 64, 64, 32, 64], "q_mode": "fp16", "runtime": "trt", "builder_policy": "trt85_default", "builder_optimization_level": None, "checkpoint_sha256": "a6015df1acb9b31c8fb975ca1b36bf5c558a965cc7635754c6b73ea1e45cbdba", "onnx_sha256": "52865e600e7d38e81e85c6d3df96e5169cdb2e19c9c64f547b963d7e28559b96"},
    "joint_fp16_control": {"widths": [32, 32, 64, 32, 64], "q_mode": "fp16", "runtime": "trt", "builder_policy": "trt85_default", "builder_optimization_level": None, "joint_cross_precision_control": True, "checkpoint_sha256": "66ff9a6bc2c5b99866d436daf7835c14707c0b54ed69e7d3ae38abf1a4f4ed3d", "onnx_sha256": "3854d04fc855b142203971273cc923522bc6ebb6da1472993dcb19240695cda5"},
}


class ManifestError(RuntimeError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _failure(status: str, message: str) -> None:
    raise ManifestError(status, message)


def _safe_write_json(path: Path, value: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        pass


def _artifact_dir(root: Path, arm: dict[str, Any]) -> Path:
    return root / "search_artifacts/sources" / "x".join(str(width) for width in arm["widths"])


def _find_one(directory: Path, pattern: str, label: str, expected_sha: str | None = None) -> Path:
    matches = sorted(directory.glob(pattern))
    if expected_sha is not None:
        matching = [path for path in matches if sha256_file(path) == expected_sha]
        if matches and not matching:
            _failure("source_sha_mismatch", f"{label} SHA mismatch in {directory}")
        matches = matching
    if not matches:
        _failure("source_artifacts_not_restored", f"missing {label} in {directory}")
    if len(matches) != 1:
        _failure("source_sha_mismatch", f"expected exactly one {label} in {directory}")
    return matches[0]


def _value(report: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in report:
            return report[name]
    return None


def _onnx_output_contract(path: Path) -> dict[str, Any]:
    try:
        import onnx
    except ImportError as error:
        _failure("source_sha_mismatch", f"onnx package required to validate frozen output: {error}")
    model = onnx.load(str(path))
    if len(model.graph.output) != 1:
        _failure("source_sha_mismatch", f"ONNX must expose one output: {path}")
    output = model.graph.output[0]
    dimensions = [dim.dim_value if dim.dim_value > 0 else None for dim in output.type.tensor_type.shape.dim]
    if not dimensions:
        _failure("source_sha_mismatch", f"ONNX output shape is absent: {path}")
    dtype = "float32" if output.type.tensor_type.elem_type == onnx.TensorProto.FLOAT else str(output.type.tensor_type.elem_type)
    if dtype != "float32":
        _failure("source_sha_mismatch", f"ONNX output dtype is not float32: {path}")
    return {"name": output.name, "shape": dimensions, "dtype": dtype}


def _validate_export_report(report_path: Path, arm: dict[str, Any], config_sha: str, checkpoint_sha: str, onnx_sha: str | None, onnx_output: dict[str, Any]) -> dict[str, Any]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _failure("source_sha_mismatch", f"invalid export report {report_path}: {error}")
    if report.get("status") != "success" or report.get("formal_measurement_eligible") is not True:
        _failure("source_sha_mismatch", f"export report is not eligible: {report_path}")
    if _value(report, "widths", "width") != arm["widths"]:
        _failure("source_sha_mismatch", f"width contract drift: {report_path}")
    if _value(report, "input_shape") != INPUT_SHAPE or _value(report, "input_name") not in (None, INPUT_NAME):
        _failure("source_sha_mismatch", f"input contract drift: {report_path}")
    pairs = (("checkpoint", checkpoint_sha), ("config", config_sha), ("onnx", onnx_sha))
    for label, actual in pairs:
        if actual is None:
            continue
        expected = _value(report, f"{label}_sha256", f"{label}_sha")
        if expected != actual:
            _failure("source_sha_mismatch", f"{label} SHA contract drift: {report_path}")
    outputs = report.get("outputs")
    if isinstance(outputs, list) and len(outputs) == 1:
        reported_name, shape = outputs[0].get("name"), outputs[0].get("shape")
    else:
        reported_name, shape = OUTPUT_NAME, report.get("output_shape")
    if reported_name != onnx_output["name"] or not isinstance(shape, list) or len(shape) != len(onnx_output["shape"]) or any(frozen is not None and frozen != reported for frozen, reported in zip(onnx_output["shape"], shape)):
        _failure("source_sha_mismatch", f"output contract drift: {report_path}")
    if not isinstance(shape, list) or not shape or any(not isinstance(value, int) or value <= 0 for value in shape):
        _failure("source_sha_mismatch", f"non-static output shape: {report_path}")
    return {"name": onnx_output["name"], "shape": shape, "dtype": onnx_output["dtype"], "raw_onnx_output_shape": onnx_output["shape"], "output_contract_source": "source_export_report" if any(value is None for value in onnx_output["shape"]) else "onnx", "requires_trt_static_resolution": any(value is None for value in onnx_output["shape"])}


def _reject_forbidden_files(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        forbidden = path.suffix.lower() == ".engine" or ("timing" in name and "cache" in name) or ("calibration" in name and "cache" in name) or ("h800" in name and ("engine" in name or "cache" in name))
        if forbidden:
            _failure("source_sha_mismatch", f"forbidden source artifact: {path}")


def _validate_global_identities(root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for relative, expected in GLOBAL_IDENTITIES.items():
        path = root / relative
        if not path.is_file():
            _failure("source_artifacts_not_restored", f"missing global artifact: {path}")
        actual = sha256_file(path)
        if actual != expected:
            _failure("source_sha_mismatch", f"global SHA mismatch: {path}")
        result[relative] = {"path": str(path.resolve()), "sha256": actual}
    for relative in REQUIRED_CONTRACT_FILES:
        path = root / relative
        if not path.is_file():
            _failure("source_artifacts_not_restored", f"missing required contract: {path}")
        result[relative] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return result


def _write_sums(root: Path, output: Path, sums_path: Path) -> None:
    records = []
    sums_resolved = sums_path.resolve()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.resolve() != sums_resolved:
            records.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    sums_path.parent.mkdir(parents=True, exist_ok=True)
    sums_path.write_text("\n".join(records) + ("\n" if records else ""), encoding="utf-8")


def create_manifest(source_root: Path | str, output: Path | str, sha_sums: Path | str) -> dict[str, Any]:
    root, output_path, sums_path = Path(source_root), Path(output), Path(sha_sums)
    receipt: dict[str, Any] = {"status": "source_artifacts_not_restored", "source_root": str(root.resolve())}
    try:
        if not root.is_dir():
            _failure("source_artifacts_not_restored", f"missing source root: {root}")
        _reject_forbidden_files(root)
        globals_ = _validate_global_identities(root)
        arms: dict[str, Any] = {}
        for name, frozen in FROZEN_ARMS.items():
            source = _artifact_dir(root, frozen)
            if not source.is_dir():
                _failure("source_artifacts_not_restored", f"missing source arm: {source}")
            config, checkpoint = source / "config.yaml", source / "recovered_checkpoint.pth"
            if not config.is_file() or not checkpoint.is_file():
                _failure("source_artifacts_not_restored", f"missing config or checkpoint in {source}")
            onnx = _find_one(source, "fcooper_dense_*.onnx", "F-Cooper ONNX", frozen.get("onnx_sha256")) if "onnx_sha256" in frozen else _find_one(source, "fcooper_dense_*.onnx", "F-Cooper ONNX")
            report_path = source / "source_export_report.json"
            if not report_path.is_file():
                _failure("source_artifacts_not_restored", f"missing export report: {report_path}")
            config_sha, checkpoint_sha = sha256_file(config), sha256_file(checkpoint)
            onnx_sha = sha256_file(onnx) if onnx is not None else None
            if checkpoint_sha != frozen["checkpoint_sha256"] or ("onnx_sha256" in frozen and onnx_sha != frozen["onnx_sha256"]):
                _failure("source_sha_mismatch", f"arm SHA mismatch: {name}")
            if name == "original_default" and config_sha != ORIGINAL_CONFIG_SHA256:
                _failure("source_sha_mismatch", f"original config SHA mismatch: {config}")
            output_contract = _validate_export_report(report_path, frozen, config_sha, checkpoint_sha, onnx_sha, _onnx_output_contract(onnx))
            arm = dict(frozen)
            arm.update({"config_path": str(config.resolve()), "checkpoint_path": str(checkpoint.resolve()), "source_export_report": str(report_path.resolve()), "config_sha256": config_sha, "checkpoint_sha256": checkpoint_sha, "input_name": INPUT_NAME, "input_shape": INPUT_SHAPE, "output_name": output_contract["name"], "output_shape": output_contract["shape"], "output_dtype": output_contract["dtype"], "raw_onnx_output_shape": output_contract["raw_onnx_output_shape"], "output_contract_source": output_contract["output_contract_source"], "requires_trt_static_resolution": output_contract["requires_trt_static_resolution"]})
            if onnx is not None and onnx_sha is not None:
                arm.update({"onnx_path": str(onnx.resolve()), "onnx_sha256": onnx_sha})
            arms[name] = arm
            if name == "original_default":
                globals_["original_checkpoint"] = {"path": str(checkpoint.resolve()), "sha256": checkpoint_sha}
                globals_["original_config"] = {"path": str(config.resolve()), "sha256": config_sha}
        receipt = {"status": "ready", "source_root": str(root.resolve()), "scope": SCOPE, "input_name": INPUT_NAME, "input_shape": INPUT_SHAPE, "output_name": OUTPUT_NAME, "global_identities": globals_, "arms": arms, "tensorrt_source_restrictions": {"platform": "Linux aarch64 Orin", "pycuda": "forbidden", "onnxruntime": "forbidden", "h800_engine_or_cache_read": False, "calibration_cache_read": False}}
        _safe_write_json(output_path, receipt)
        _write_sums(root, output_path, sums_path)
        return receipt
    except ManifestError as error:
        receipt.update({"status": error.status, "error": str(error)})
        _safe_write_json(output_path, receipt)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--source-root", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--sha-sums", required=True)
    args = parser.parse_args(argv)
    try:
        create_manifest(args.source_root, args.output, args.sha_sums)
    except ManifestError as error:
        print(f"{error.status}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
