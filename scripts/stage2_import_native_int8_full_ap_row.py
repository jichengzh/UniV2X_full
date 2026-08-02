#!/usr/bin/env python3
"""Import a gated native-INT8 full AP eval report as an original60 AP row."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import (  # noqa: E402
    ap_anchor_row,
    append_jsonl,
    parse_width_csv,
    stable_config_id,
    utc_timestamp,
    validate_lut_row,
)
from scripts.stage2_h800_true_fp16_ap_eval import write_json  # noqa: E402


DEFAULT_ROWS_OUT = (
    ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/"
    "rows/native_int8_original60_ap_rows_v1.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--width", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--route-dir", required=True)
    parser.add_argument("--tensor-quant-params-path", required=True)
    parser.add_argument("--rows-out", default=str(DEFAULT_ROWS_OUT))
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--created-at", default=None)
    parser.add_argument("--dataset", default="DAIR-V2X")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--min-samples", type=int, default=1789)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def output_dequant_schemes(output_dequant_summary: dict[str, Any]) -> dict[str, str]:
    outputs = output_dequant_summary.get("outputs")
    if not isinstance(outputs, list):
        outputs = output_dequant_summary.get("dequant_outputs")
    if not isinstance(outputs, list):
        outputs = output_dequant_summary.get("items")
    if not isinstance(outputs, list):
        outputs = []
    schemes: dict[str, str] = {}
    for item in outputs:
        if not isinstance(item, dict):
            continue
        name = str(item.get("tensor_name") or item.get("arg_name") or "")
        scheme = str(item.get("scheme") or item.get("dequant_scheme") or "")
        if name:
            schemes[name] = scheme
    return schemes


def validate_report_gate(
    report: dict[str, Any],
    *,
    output_dequant_summary: dict[str, Any],
    min_samples: int,
) -> None:
    processed = int(report.get("processed_samples") or report.get("num_samples") or 0)
    if processed < int(min_samples):
        raise RuntimeError(f"refusing native INT8 AP row for partial eval: processed_samples={processed}")
    if not bool(report.get("ap_row_allowed")):
        raise RuntimeError(f"native INT8 AP row gate is not allowed: {report.get('ap_row_block_reason')}")
    if int(report.get("ap_row_min_samples") or 0) < int(min_samples):
        raise RuntimeError(f"native INT8 AP row min-sample gate is too low: {report.get('ap_row_min_samples')}")
    if int(report.get("pred_nonempty_count") or 0) <= 0:
        raise RuntimeError("native INT8 AP row requires nonempty predictions")
    for key in ("ap30", "ap50", "ap70"):
        value = report.get(key)
        if not isinstance(value, (int, float)):
            raise RuntimeError(f"native INT8 AP row requires finite numeric {key}: {value}")
    schemes = output_dequant_schemes(output_dequant_summary)
    required_outputs = {"pyramid_level0", "pyramid_level1", "pyramid_level2"}
    missing = sorted(required_outputs - set(schemes))
    if missing:
        raise RuntimeError(f"output_dequant_summary missing outputs: {missing}")
    bad = {name: schemes[name] for name in sorted(required_outputs) if schemes[name] != "tensor_quant_params_v2"}
    if bad:
        raise RuntimeError(f"native INT8 AP row requires tensor_quant_params_v2 dequant: {bad}")


def build_row(args: argparse.Namespace, report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    label = str(args.label)
    width = parse_width_csv(args.width)
    raw_dir = Path(args.raw_dir)
    route_dir = Path(args.route_dir)
    tensor_quant_path = Path(args.tensor_quant_params_path)
    output_dequant_path = raw_dir / "output_dequant_summary.json"
    worker_response_summary = raw_dir / "worker_response_summary.json"
    postprocess_summary = raw_dir / "postprocess_summary.json"
    route_manifest = route_dir / "native_int8_route_manifest.json"
    output_dequant_summary = read_json(output_dequant_path)
    validate_report_gate(
        report,
        output_dequant_summary=output_dequant_summary,
        min_samples=int(args.min_samples),
    )

    config_id = stable_config_id(
        model="pyramid_lidar",
        candidate_id=f"original60:{label}:native_int8:ap",
        software_point_id=f"native_int8_original60:{label}:{'x'.join(str(item) for item in width)}:ap",
        quant_policy="int8",
        schedule_policy="not_applicable",
    )
    source_files = [
        str(report_path),
        str(output_dequant_path),
        str(tensor_quant_path),
    ]
    for path in (worker_response_summary, postprocess_summary, route_manifest):
        if path.exists():
            source_files.append(str(path))
    row = ap_anchor_row(
        config_id=config_id,
        model="pyramid_lidar",
        manifest_digest=sha256_file(route_manifest) if route_manifest.exists() else sha256_file(tensor_quant_path),
        candidate_id=f"original60:pyramid_lidar:{label}:native_int8",
        software_point_id=f"native_int8_original60:{label}:{'x'.join(str(item) for item in width)}:ap",
        dense_stage="model",
        optimized_scope="native_int8_backbone_torch_head_ap_eval",
        width=width,
        quant_policy="int8",
        schedule_policy="not_applicable",
        backend="model_eval",
        measurement_status="measured",
        metric="AP70",
        metric_value=float(report["ap70"]),
        secondary_metrics={"AP30": float(report["ap30"]), "AP50": float(report["ap50"])},
        dataset=str(report.get("dataset") or args.dataset),
        eval_split=str(report.get("eval_split") or args.eval_split),
        num_samples=int(report.get("processed_samples") or report.get("num_samples") or 0),
        pred_nonempty_count=int(report.get("pred_nonempty_count") or 0),
        pred_total_count=int(report.get("pred_total_count") or 0),
        native_int8_route_dir=str(route_dir),
        native_int8_route_manifest=str(route_manifest) if route_manifest.exists() else "",
        tensor_quant_params_path=str(tensor_quant_path),
        tensor_quant_params_digest=sha256_file(tensor_quant_path),
        output_dequant_summary=str(output_dequant_path),
        output_dequant_policy=str(report.get("output_dequant_policy") or "per_output_tensor_quant_params_v2"),
        worker_response_summary=str(worker_response_summary) if worker_response_summary.exists() else "",
        postprocess_summary=str(postprocess_summary) if postprocess_summary.exists() else "",
        ckpt_path=str(report.get("ckpt_path") or ""),
        ckpt_digest=(
            sha256_file(Path(str(report.get("ckpt_path"))))
            if str(report.get("ckpt_path") or "") and Path(str(report.get("ckpt_path"))).exists()
            else "unknown"
        ),
        eval_command=str(report.get("eval_command") or "stage2_h800_native_int8_real_activation_bridge.py"),
        provenance="stage2_original60_native_int8_h800_full_ap_bridge_eval",
        run_id=args.run_id or f"native_int8_original60_full_ap_{label}",
        created_at=args.created_at or utc_timestamp(),
        source_files=source_files,
        raw_artifact=str(raw_dir),
        failure_reason=None,
        notes="native INT8 TVM backbone/subnet full AP bridge row from gated 1789-sample eval",
        precision="int8",
        quant_scheme="tvm_native_int8_backbone_subnet_activation_calibration_v2",
        quant_method="h800_tvm_native_int8_backbone_subnet",
        quant_scope="backbone_subnet_native_int8",
        calibration_source="tensor_quant_params_calibration_v2",
        calibration_digest=sha256_file(tensor_quant_path),
        calibrator="reference_range_capture_pyramid_level2",
        calibration_inputs=["spatial_features", "pyramid_level0", "pyramid_level1", "pyramid_level2"],
        fallback_policy="none_for_graph_outputs",
        layer_precision_summary=str(output_dequant_path),
        full_network_claim=False,
        engine_kind="tvm_graph_executor",
        engine_digest=sha256_file(report_path),
        measurement_source="true_eval",
        claim_status="claimable_native_int8_full_ap_eval",
        quality_gate_status="native_int8_full_ap_eval",
        schedule_profile="not_applicable",
        tune_budget="native_int8_route_existing",
        label=label,
        original60_candidate_id=f"original60:{label}",
    )
    validate_lut_row(row)
    return row


def write_blocker(raw_dir: Path, args: argparse.Namespace, reason: str, traceback_text: str = "") -> None:
    write_json(
        raw_dir / "native_int8_ap_row_import_blocker.json",
        {
            "schema": "native_int8_ap_row_import_blocker_v1",
            "label": args.label,
            "precision": "native_int8",
            "failure_reason": reason,
            "traceback": traceback_text,
            "raw_artifact": str(raw_dir),
            "command": sys.argv,
            "created_at": args.created_at or utc_timestamp(),
        },
    )


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    report_path = Path(args.report_json) if args.report_json else raw_dir / "full_ap_eval_report.json"
    try:
        report = read_json(report_path)
        row = build_row(args, report, report_path)
        append_jsonl(args.rows_out, row)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "row_id": row["row_id"],
                    "rows_out": str(args.rows_out),
                    "report_json": str(report_path),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        reason = f"{type(exc).__name__}:{exc}"
        write_blocker(raw_dir, args, reason, traceback.format_exc())
        print(json.dumps({"status": "failed", "failure_reason": reason}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
