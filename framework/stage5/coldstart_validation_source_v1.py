"""Rebind frozen Gold176 genomes to their original executable source artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from framework.stage5.measurement_plan_v1 import _source_plan_sha


EVIDENCE_SCHEMA = "stage5_source_materialization_evidence_v1"
CATALOG_SCHEMA = "stage5_coldstart_validation_source_catalog_v1"


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one(paths: Sequence[Path], label: str) -> Path:
    matches = [path for path in paths if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"expected one {label}, got {len(matches)}")
    return matches[0]


def _historical_source(repo: Path, model: str, width: Sequence[int]) -> dict[str, Any]:
    key = "x".join(map(str, width))
    padded = "x".join(f"{value:03d}" for value in width)
    if model == "codriving":
        root = Path(
            "/exdata/jichengzhi/V2Xverse_pyramid/output/"
            "codriving_v2_gold_ap_20260709"
        ) / key
        checkpoint = _one(sorted(root.glob("net_epoch_bestval_at*.pth")), "CoDriving checkpoint")
        return {
            "materialization_kind": "codriving_prepare_train_export",
            "checkpoint_path": checkpoint,
            "checkpoint_dir": root,
            "onnx_path": root / f"resnet_multiscale_{key}_final_fp32.onnx",
            "calibration_npz": root / "stage3_calib_train_n16_float32.npz",
            "calibration_root": root / "calibration_source",
            "calibration_summary": root / "stage3_calib_train_n16_float32_summary.json",
            "trt_calibration_dir": root / "trt_calibration_npy",
        }
    if model != "pyramid":
        raise ValueError(f"unsupported cold-start model: {model}")
    if list(width) == [16, 32, 64]:
        root = repo / "results/stage35_gold32_supplement_v1_20260713"
        checkpoint_dir = Path(
            "/home/jichengzhi/heal_research/checkpoints/stage1/"
            "Pyramid_DAIR_m1_pruned75_2026_05_10"
        )
        checkpoint = checkpoint_dir / "net_epoch_bestval_at31.pth"
        source_dir = root / "pyramid_sources_shape_repaired_v1" / padded
        calibration_dir = root / "pyramid_calibration" / padded
    else:
        root = repo / "results/stage3_gold96_v3_20260711"
        source_dir = root / "pyramid_sources" / padded
        calibration_dir = root / "pyramid_calibration" / padded
        report = json.loads((source_dir / "export_report.json").read_text(encoding="utf-8"))
        checkpoint = Path(
            str(report.get("checkpoint_path") or report.get("ckpt_path") or "")
        )
        checkpoint_dir = checkpoint.parent
    return {
        "materialization_kind": "pyramid_checkpoint_export",
        "checkpoint_path": checkpoint,
        "checkpoint_dir": checkpoint_dir,
        "onnx_path": source_dir / f"pyramid_{padded}_multiscale.onnx",
        "calibration_npz": calibration_dir / "spatial_features_train16.npz",
        "calibration_root": calibration_dir,
        "calibration_summary": calibration_dir / "summary.json",
        "trt_calibration_dir": calibration_dir / "trt_npy",
    }


def quant_contract_from_gold_result(source_gold: Mapping[str, Any]) -> Path | None:
    """Recover the exact quant contract used by the Gold performance artifact."""
    result_path = Path(str(source_gold.get("performance_result_json") or ""))
    if not result_path.is_file():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    quant_path = Path(str(result.get("tensor_quant_params_path") or ""))
    if not quant_path.is_file():
        return None
    expected = str(result.get("tensor_quant_params_sha256") or "")
    if len(expected) != 64 or _file_sha(quant_path) != expected:
        raise ValueError(f"Gold quant contract SHA mismatch: {quant_path}")
    return quant_path


def _quant_contract(
    repo: Path,
    model: str,
    width: Sequence[int],
    source_gold: Mapping[str, Any],
) -> Path:
    bound = quant_contract_from_gold_result(source_gold)
    if bound is not None:
        return bound
    key = "x".join(map(str, width))
    if list(width) == [16, 32, 64] and model == "codriving":
        return repo / f"results/stage3_tvm_int8_repair_v3_20260713/{model}/{key}/tensor_quant_params.json"
    return repo / (
        f"results/stage35_gold32_supplement_v1_20260713/ap_v1/"
        f"tvm_int8_repair/{model}/{key}/tensor_quant_params.json"
    )


def build_coldstart_fallback(
    *,
    repo_root: Path,
    output_dir: Path,
    gold_rows: Sequence[Mapping[str, Any]],
    selected_ids: Sequence[str],
    task_template: Mapping[str, Any],
) -> dict[str, Any]:
    """Create executable rows only for frozen IDs absent from online requests."""
    gold = {str(row.get("manifest_job_id") or ""): row for row in gold_rows}
    rows: dict[str, dict[str, Any]] = {}
    evidence_paths: dict[str, Path] = {}
    quant_paths: dict[str, Path] = {}
    provenance = []
    source_root = output_dir / "coldstart_sources"
    source_root.mkdir(parents=True, exist_ok=True)
    for row_id in selected_ids:
        source_gold = gold.get(row_id)
        if source_gold is None:
            continue
        model = str(source_gold["model"])
        width = [int(value) for value in source_gold["width"]]
        group_id = f"{model}|{'x'.join(map(str, width))}"
        resolved = _historical_source(repo_root, model, width)
        materialization_kind = str(resolved.pop("materialization_kind"))
        marker = source_root / f"{model}_{'_'.join(map(str, width))}.done"
        contract = {key: str(value) for key, value in resolved.items()}
        contract["source_done_marker"] = str(marker)
        row = {
            "row_id": row_id,
            "manifest_job_id": row_id,
            "task_id": str(task_template["task_id"]),
            "task_sha256": str(task_template["task_sha256"]),
            "group_id": group_id,
            "model": model,
            "width": width,
            "genome": [*width, str(source_gold["q_mode"])],
            "q_mode": str(source_gold["q_mode"]),
            "hardware_id": str(task_template["hardware_id"]),
            "capability_profile_id": str(source_gold["capability_profile_id"]),
            "capability_digest": str(task_template["capability_digest"]),
            "dispatch_key": str(source_gold["dispatch_key"]),
            "materialization_kind": materialization_kind,
            "source_contract": contract,
            "validation_origin": "gold176_coldstart",
        }
        row["source_evidence_sha256"] = _source_plan_sha(row)
        artifacts = {
            "checkpoint": Path(contract["checkpoint_path"]),
            "onnx": Path(contract["onnx_path"]),
            "calibration": Path(contract["calibration_npz"]),
            "calibration_summary": Path(contract["calibration_summary"]),
        }
        for label, path in artifacts.items():
            if not path.is_file():
                raise ValueError(f"historical {label} missing for {row_id}: {path}")
        trt_dir = Path(contract["trt_calibration_dir"])
        if not trt_dir.is_dir() or not any(trt_dir.glob("*.npy")):
            raise ValueError(f"historical TRT calibration missing for {row_id}: {trt_dir}")
        evidence = {
            "schema_version": EVIDENCE_SCHEMA,
            "group_id": group_id,
            "source_plan_sha256": row["source_evidence_sha256"],
            "status": "ready",
            "checkpoint_path": str(artifacts["checkpoint"]),
            "checkpoint_sha256": _file_sha(artifacts["checkpoint"]),
            "onnx_path": str(artifacts["onnx"]),
            "onnx_sha256": _file_sha(artifacts["onnx"]),
            "calibration_path": str(artifacts["calibration"]),
            "calibration_sha256": _file_sha(artifacts["calibration"]),
            "calibration_summary_path": str(artifacts["calibration_summary"]),
            "calibration_summary_sha256": _file_sha(artifacts["calibration_summary"]),
        }
        evidence_path = marker.with_name(marker.stem + "_evidence.json")
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        marker.touch()
        rows[row_id] = row
        evidence_paths[group_id] = evidence_path
        if row["dispatch_key"] == "tvm_auto" and row["q_mode"] == "int8":
            quant = _quant_contract(repo_root, model, width, source_gold)
            if not quant.is_file():
                raise ValueError(f"historical quant contract missing for {row_id}: {quant}")
            quant_paths[row_id] = quant
        provenance.append(
            {
                "manifest_job_id": row_id,
                "gold_performance_result_sha256": source_gold.get("performance_result_sha256"),
                "gold_ap_report_sha256": source_gold.get("ap_report_sha256"),
                "source_evidence_path": str(evidence_path),
                "source_evidence_file_sha256": _file_sha(evidence_path),
                "quant_contract_path": str(quant_paths.get(row_id, "")),
            }
        )
    return {
        "schema_version": CATALOG_SCHEMA,
        "rows": rows,
        "source_evidence_paths": evidence_paths,
        "quant_contract_paths": quant_paths,
        "provenance": provenance,
    }


__all__ = [
    "build_coldstart_fallback",
    "quant_contract_from_gold_result",
    "CATALOG_SCHEMA",
]
