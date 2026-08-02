#!/usr/bin/env python3
"""Prepare held-out Pyramid inputs and compare three-level backbone outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def prepare_heldout_batches(
    features: np.ndarray,
    scene_record_lens: list[int],
    *,
    calibration_scene_count: int,
) -> tuple[np.ndarray, dict[str, int]]:
    array = np.asarray(features)
    if array.ndim != 4:
        raise ValueError(f"features must be [agents,C,H,W], got {list(array.shape)}")
    if sum(scene_record_lens) != array.shape[0]:
        raise ValueError("scene record lengths do not cover the feature agent axis")
    if not 0 < calibration_scene_count < len(scene_record_lens):
        raise ValueError("calibration scene count must split the exported scenes")
    excluded_agents = int(sum(scene_record_lens[:calibration_scene_count]))
    heldout = array[excluded_agents:]
    used_agents = int((heldout.shape[0] // 2) * 2)
    if used_agents == 0:
        raise ValueError("held-out scene suffix contains no complete batch2 input")
    batches = np.ascontiguousarray(
        heldout[:used_agents].reshape(-1, 2, *heldout.shape[1:])
    )
    return batches, {
        "excluded_scene_count": int(calibration_scene_count),
        "heldout_scene_count": int(len(scene_record_lens) - calibration_scene_count),
        "excluded_agent_instances": excluded_agents,
        "heldout_agent_instances_available": int(heldout.shape[0]),
        "heldout_agent_instances_used": used_agents,
        "heldout_agent_instances_dropped": int(heldout.shape[0] - used_agents),
        "heldout_batch2_count": int(batches.shape[0]),
    }


def compute_output_metrics(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape:
        raise ValueError(f"output shape mismatch: {list(ref.shape)} != {list(cand.shape)}")
    finite = bool(np.isfinite(ref).all() and np.isfinite(cand).all())
    if not finite:
        raise ValueError("reference or candidate output contains non-finite values")
    ref_flat = ref.reshape(-1)
    cand_flat = cand.reshape(-1)
    difference = cand_flat - ref_flat
    ref_rms = float(np.sqrt(np.mean(np.square(ref_flat))))
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    denominator = float(np.linalg.norm(ref_flat) * np.linalg.norm(cand_flat))
    cosine = (
        float(np.dot(ref_flat, cand_flat) / denominator)
        if denominator > 0.0
        else float(ref_flat.size == 0 or np.array_equal(ref_flat, cand_flat))
    )
    ref_min = float(np.min(ref_flat))
    ref_max = float(np.max(ref_flat))
    cand_min = float(np.min(cand_flat))
    cand_max = float(np.max(cand_flat))
    abs_ref_p999 = float(np.quantile(np.abs(ref_flat), 0.999))
    return {
        "shape": list(ref.shape),
        "finite": finite,
        "cosine": cosine,
        "nrmse": float(rmse / max(ref_rms, 1e-12)),
        "rmse": rmse,
        "mae": float(np.mean(np.abs(difference))),
        "reference_min": ref_min,
        "reference_max": ref_max,
        "candidate_min": cand_min,
        "candidate_max": cand_max,
        "reference_zero_fraction": float(np.mean(ref_flat == 0.0)),
        "candidate_zero_fraction": float(np.mean(cand_flat == 0.0)),
        "candidate_at_observed_min_fraction": float(np.mean(cand_flat == cand_min)),
        "candidate_at_observed_max_fraction": float(np.mean(cand_flat == cand_max)),
        "candidate_below_reference_min_fraction": float(np.mean(cand_flat < ref_min)),
        "candidate_above_reference_max_fraction": float(np.mean(cand_flat > ref_max)),
        "reference_abs_p999": abs_ref_p999,
        "candidate_abs_ge_reference_p999_fraction": float(
            np.mean(np.abs(cand_flat) >= abs_ref_p999)
        ),
        "saturation_clipping_semantics": (
            "output_distribution_proxy_not_internal_quantizer_counter"
        ),
    }


def _load_spatial_features(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "spatial_features" not in archive:
            raise KeyError(f"{path} has no spatial_features array")
        return np.asarray(archive["spatial_features"])


def command_prepare(args: argparse.Namespace) -> int:
    summary = json.loads(args.export_summary.read_text(encoding="utf-8"))
    scene_lens = [int(value) for value in summary["scene_record_lens"]]
    all_features = _load_spatial_features(args.export_npz)
    calibration_features = _load_spatial_features(args.calibration_npz)
    excluded_agents = sum(scene_lens[: args.calibration_scene_count])
    prefix = all_features[:excluded_agents]
    if prefix.shape != calibration_features.shape:
        raise ValueError(
            f"calibration prefix shape mismatch: {list(prefix.shape)} != "
            f"{list(calibration_features.shape)}"
        )
    prefix_identical = bool(np.array_equal(prefix, calibration_features))
    if not prefix_identical and not args.allow_nondeterministic_prefix:
        raise ValueError(
            "calibration prefix is not byte-identical; pass "
            "--allow-nondeterministic-prefix only for explicitly degraded evidence"
        )
    batches, split_audit = prepare_heldout_batches(
        all_features,
        scene_lens,
        calibration_scene_count=args.calibration_scene_count,
    )
    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, batches)
    audit = {
        "schema_version": "lane_c_backbone_heldout_split_v1",
        "dataset": "DAIR-V2X-C frozen train split",
        "heldout_definition": (
            "deterministic scene suffix after the first 16 calibration scenes"
        ),
        "export_npz": str(args.export_npz),
        "export_npz_sha256": sha256_file(args.export_npz),
        "export_summary": str(args.export_summary),
        "export_summary_sha256": sha256_file(args.export_summary),
        "calibration_npz": str(args.calibration_npz),
        "calibration_npz_sha256": sha256_file(args.calibration_npz),
        "calibration_prefix_array_sha256": sha256_array(prefix),
        "calibration_reference_array_sha256": sha256_array(calibration_features),
        "calibration_prefix_array_identical": prefix_identical,
        "calibration_prefix_mismatch_explicitly_allowed": bool(
            args.allow_nondeterministic_prefix
        ),
        "output_npy": str(args.output_npy),
        "output_npy_sha256": sha256_file(args.output_npy),
        "output_shape": list(batches.shape),
        **split_audit,
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


def command_ort_reference(args: argparse.Namespace) -> int:
    import onnxruntime as ort

    inputs = np.load(args.inputs_npy, allow_pickle=False).astype(np.float32)
    session = ort.InferenceSession(
        str(args.onnx), providers=["CPUExecutionProvider"]
    )
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 3:
        raise ValueError("reference ONNX must expose one input and three outputs")
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    accumulated = {name: [] for name in output_names}
    for batch in inputs:
        values = session.run(output_names, {input_name: batch})
        for name, value in zip(output_names, values):
            accumulated[name].append(np.asarray(value, dtype=np.float32))
    outputs = {name: np.stack(values) for name, values in accumulated.items()}
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **outputs)
    report = {
        "schema_version": "lane_c_backbone_server_ort_reference_v1",
        "execution_provider": "CPUExecutionProvider",
        "onnx_sha256": sha256_file(args.onnx),
        "inputs_sha256": sha256_file(args.inputs_npy),
        "output_npz": str(args.output_npz),
        "output_npz_sha256": sha256_file(args.output_npz),
        "output_shapes": {name: list(value.shape) for name, value in outputs.items()},
        "all_finite": all(bool(np.isfinite(value).all()) for value in outputs.values()),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def command_compare(args: argparse.Namespace) -> int:
    with np.load(args.reference_npz, allow_pickle=False) as reference_archive:
        reference = {name: reference_archive[name] for name in reference_archive.files}
    with np.load(args.candidate_npz, allow_pickle=False) as candidate_archive:
        candidate = {name: candidate_archive[name] for name in candidate_archive.files}
    if set(reference) != set(candidate):
        raise ValueError(
            f"output names differ: {sorted(reference)} != {sorted(candidate)}"
        )
    per_output = {}
    for name in sorted(reference):
        reference_dtype = reference[name].dtype
        candidate_dtype = candidate[name].dtype
        if args.precision == "fp32":
            for source, dtype in (
                ("reference", reference_dtype),
                ("candidate", candidate_dtype),
            ):
                if dtype != np.dtype(np.float32):
                    raise ValueError(
                        "fp32 comparison requires float32 "
                        f"{source} output {name}, got {dtype}"
                    )
        per_output[name] = {
            **compute_output_metrics(reference[name], candidate[name]),
            "reference_dtype": str(reference_dtype),
            "candidate_dtype": str(candidate_dtype),
        }
    report = {
        "schema_version": "lane_c_backbone_cross_hardware_numerical_v1",
        "precision": args.precision,
        "dtype_contract": {
            "required_dtype": "float32" if args.precision == "fp32" else None,
            "validated": args.precision == "fp32",
        },
        "reference_npz": str(args.reference_npz),
        "reference_npz_sha256": sha256_file(args.reference_npz),
        "candidate_npz": str(args.candidate_npz),
        "candidate_npz_sha256": sha256_file(args.candidate_npz),
        "per_output": per_output,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _require_within(root: Path, path: Path, *, label: str) -> None:
    if path != root and root not in path.parents:
        raise ValueError(f"{label} must remain under artifact root {root}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-heldout")
    prepare.add_argument("--artifact-root", type=_path, required=True)
    prepare.add_argument("--export-npz", type=_path, required=True)
    prepare.add_argument("--export-summary", type=_path, required=True)
    prepare.add_argument("--calibration-npz", type=_path, required=True)
    prepare.add_argument("--calibration-scene-count", type=int, default=16)
    prepare.add_argument("--allow-nondeterministic-prefix", action="store_true")
    prepare.add_argument("--output-npy", type=_path, required=True)
    prepare.add_argument("--audit-json", type=_path, required=True)
    prepare.set_defaults(handler=command_prepare)

    reference = commands.add_parser("ort-reference")
    reference.add_argument("--artifact-root", type=_path, required=True)
    reference.add_argument("--onnx", type=_path, required=True)
    reference.add_argument("--inputs-npy", type=_path, required=True)
    reference.add_argument("--output-npz", type=_path, required=True)
    reference.add_argument("--report-json", type=_path, required=True)
    reference.set_defaults(handler=command_ort_reference)

    compare = commands.add_parser("compare")
    compare.add_argument("--artifact-root", type=_path, required=True)
    compare.add_argument("--reference-npz", type=_path, required=True)
    compare.add_argument("--candidate-npz", type=_path, required=True)
    compare.add_argument("--precision", choices=("fp32", "fp16", "int8"), required=True)
    compare.add_argument("--report-json", type=_path, required=True)
    compare.set_defaults(handler=command_compare)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_fields = {
        "prepare-heldout": ("output_npy", "audit_json"),
        "ort-reference": ("output_npz", "report_json"),
        "compare": ("report_json",),
    }[args.command]
    for field in output_fields:
        _require_within(args.artifact_root, getattr(args, field), label=field)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
