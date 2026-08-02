#!/usr/bin/env python3
"""Freeze the 8-group cross-model Gold32 supplement before measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage2.canonical_search_v3 import build_active_manifest


SCHEMA_VERSION = "stage35_gold32_supplement_plan_v1"
SUPPLEMENT_GROUPS = (
    ("pyramid", "16x32x64", "train", "codriving|16x32x64"),
    ("pyramid", "24x64x128", "train", "codriving|24x64x128"),
    ("pyramid", "32x64x128", "train", "codriving|32x64x128"),
    ("pyramid", "64x128x256", "locked_holdout", "codriving|64x128x256"),
    ("codriving", "16x64x128", "train", "pyramid|16x64x128"),
    ("codriving", "24x56x128", "train", "pyramid|24x56x128"),
    ("codriving", "32x32x256", "train", "pyramid|32x32x256"),
    ("codriving", "40x80x160", "locked_holdout", "pyramid|40x80x160"),
)
PYRAMID_CHECKPOINT_SOURCES = {
    "16x32x64": (
        "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10",
        "net_epoch_bestval_at31.pth",
        "pruned75",
    ),
    "24x64x128": (
        "/exdata/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_frontier_01_2026_06_28",
        "net_epoch31.pth",
        "frontier_01",
    ),
    "32x64x128": (
        "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10",
        "net_epoch_bestval_at29.pth",
        "pruned50",
    ),
    "64x128x256": (
        "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29",
        "net_epoch_bestval_at23.pth",
        "base",
    ),
}
REMOTE_REPO = Path("/home/jichengzhi/V2X")
REMOTE_RESULT_ROOT = REMOTE_REPO / "results/stage35_gold32_supplement_v1_20260713"
CODRIVING_ROOT = Path("/exdata/jichengzhi/V2Xverse_pyramid/output/codriving_v2_gold_ap_20260709")


def _sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _parse_width(width_key: str) -> list[int]:
    width = [int(part) for part in width_key.split("x")]
    if len(width) != 3 or any(value <= 0 for value in width):
        raise ValueError(f"invalid width key: {width_key}")
    return width


def _padded_width(width_key: str) -> str:
    return "x".join(f"{value:03d}" for value in _parse_width(width_key))


def _source_contract(model: str, width_key: str) -> dict[str, Any]:
    if model == "pyramid":
        padded = _padded_width(width_key)
        checkpoint_dir_value, checkpoint_name, label = PYRAMID_CHECKPOINT_SOURCES[width_key]
        checkpoint_dir = Path(checkpoint_dir_value)
        checkpoint_path = checkpoint_dir / checkpoint_name
        source_root = REMOTE_RESULT_ROOT / "pyramid_sources" / padded
        calibration_root = REMOTE_RESULT_ROOT / "pyramid_calibration" / padded
        return {
            "source_kind": "checkpoint_consistent_pyramid_multiscale",
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_glob": str(checkpoint_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_label": label,
            "onnx_path": str(source_root / f"pyramid_{padded}_multiscale.onnx"),
            "onnx_report_path": str(source_root / "onnx_export_report.json"),
            "calibration_root": str(calibration_root),
            "calibration_npz": str(calibration_root / "spatial_features_train16.npz"),
            "calibration_summary": str(calibration_root / "summary.json"),
        }
    if model == "codriving":
        model_dir = CODRIVING_ROOT / width_key
        return {
            "source_kind": "checkpoint_consistent_codriving_multiscale",
            "model_dir": str(model_dir),
            "checkpoint_glob": str(model_dir / "net_epoch_bestval_at*.pth"),
            "onnx_path": str(model_dir / f"resnet_multiscale_{width_key}_final_fp32.onnx"),
            "calibration_root": str(model_dir / "calibration_source"),
            "calibration_npz": str(model_dir / "stage3_calib_train_n16_float32.npz"),
            "calibration_summary": str(
                model_dir / "stage3_calib_train_n16_float32_summary.json"
            ),
        }
    raise ValueError(f"unsupported model: {model}")


def _validate_profiles(capability_profiles: Sequence[Mapping[str, Any]]) -> None:
    if len(capability_profiles) != 2:
        raise ValueError("Gold32 supplement requires exactly two capability profiles")
    dispatch_keys = {str(profile.get("dispatch_key")) for profile in capability_profiles}
    if dispatch_keys != {"tvm_auto", "trt_engine"}:
        raise ValueError("capability profiles must be the frozen TVM-auto/TRT pair")


def build_supplement_plan(capability_profiles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the premeasurement Gold32 supplement manifest and split."""
    _validate_profiles(capability_profiles)
    widths_by_model = {
        model: [width for row_model, width, _, _ in SUPPLEMENT_GROUPS if row_model == model]
        for model in ("pyramid", "codriving")
    }
    active = build_active_manifest(
        widths_by_model=widths_by_model,
        capability_profiles=capability_profiles,
    )
    group_meta = {
        f"{model}|{width}": {
            "split": split,
            "counterpart_group_id": counterpart,
            "source_contract": _source_contract(model, width),
        }
        for model, width, split, counterpart in SUPPLEMENT_GROUPS
    }

    jobs: list[dict[str, Any]] = []
    for source in active["jobs"]:
        row = dict(source)
        metadata = group_meta[str(row["group_id"])]
        jobs.append(
            {
                **row,
                "genome": [*row["width"], row["q_mode"]],
                "width_stratum": "cross_model_common_anchor",
                "split": metadata["split"],
                "counterpart_group_id": metadata["counterpart_group_id"],
                "source_contract": metadata["source_contract"],
                "source_status": "pending_checkpoint_consistent_source",
                "required_metrics": ["latency", "energy", "ap"],
                "terminal_status": "pending",
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in jobs:
        grouped.setdefault(str(row["group_id"]), []).append(row)
    expected_product = {
        (q_mode, dispatch_key)
        for q_mode in ("fp16", "int8")
        for dispatch_key in ("tvm_auto", "trt_engine")
    }
    if len(grouped) != 8:
        raise ValueError("supplement must contain eight model-width groups")
    for group_id, rows in grouped.items():
        actual = {(str(row["q_mode"]), str(row["dispatch_key"])) for row in rows}
        if len(rows) != 4 or actual != expected_product:
            raise ValueError(f"incomplete 2x2 arm product: {group_id}")

    train_group_ids = sorted(
        group_id for group_id, rows in grouped.items() if rows[0]["split"] == "train"
    )
    holdout_group_ids = sorted(set(grouped) - set(train_group_ids))
    if len(train_group_ids) != 6 or len(holdout_group_ids) != 2:
        raise ValueError("supplement split must be 6 train groups and 2 locked holdout groups")

    manifest = {
        **active,
        "schema_version": "stage35_gold32_supplement_manifest_v1",
        "stage35_plan_schema_version": SCHEMA_VERSION,
        "selection_basis": "cross_model_width_counterparts_selected_before_measurement",
        "jobs": jobs,
    }
    split = {
        "schema_version": "stage35_gold32_supplement_split_v1",
        "selection_basis": "inherit_gold96_counterpart_split_before_measurement",
        "train_group_ids": train_group_ids,
        "holdout_group_ids": holdout_group_ids,
        "train_job_ids": sorted(row["job_id"] for row in jobs if row["split"] == "train"),
        "holdout_job_ids": sorted(
            row["job_id"] for row in jobs if row["split"] == "locked_holdout"
        ),
    }
    contract = {
        "schema_version": "stage35_gold32_supplement_contract_v1",
        "row_count": 32,
        "group_count": 8,
        "group_size": 4,
        "train_rows": 24,
        "locked_holdout_rows": 8,
        "required_metrics": ["latency", "energy", "ap"],
        "checkpoint_consistent_source_required": True,
        "random_weight_onnx_allowed": False,
        "historical_measurement_fill_allowed": False,
        "mixed_policy_activation": "disabled",
        "failure_policy": "retry_once_then_record_feasibility_failure",
    }
    return {"manifest": manifest, "split": split, "contract": contract}


def _write_content_addressed(output_dir: Path, stem: str, payload: Mapping[str, Any]) -> Path:
    digest = _sha256(payload)
    path = output_dir / f"{stem}-{digest}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = json.loads(args.capability_manifest_json.read_text(encoding="utf-8"))
    plan = build_supplement_plan(source["capability_profiles"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _write_content_addressed(
        args.output_dir, "gold32_supplement_manifest_v1", plan["manifest"]
    )
    split_path = _write_content_addressed(
        args.output_dir, "gold32_supplement_split_v1", plan["split"]
    )
    contract_path = _write_content_addressed(
        args.output_dir, "gold32_supplement_contract_v1", plan["contract"]
    )
    index = {
        "schema_version": SCHEMA_VERSION,
        "manifest_json": str(manifest_path.resolve()),
        "split_json": str(split_path.resolve()),
        "contract_json": str(contract_path.resolve()),
        "manifest_sha256": _sha256(plan["manifest"]),
        "split_sha256": _sha256(plan["split"]),
        "contract_sha256": _sha256(plan["contract"]),
    }
    index_path = args.output_dir / "gold32_supplement_plan_v1.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**index, "index_json": str(index_path.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
