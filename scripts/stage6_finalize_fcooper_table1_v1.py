#!/usr/bin/env python3
"""Finalize audited F-Cooper five-arm results for the paper table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


SUCCESS = "measured_success_gold"


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite measured value: {value!r}")
    return number


def select_arm_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    ap70_ref: float,
    max_ap_drop: float,
) -> dict[str, Any]:
    successful = [
        {
            **dict(row),
            "ap70": _finite(row["ap70"]),
            "latency_ms": _finite(row["latency_ms"]),
            "energy_j": _finite(row["energy_j"]),
        }
        for row in rows
        if row.get("terminal_status") == SUCCESS
    ]
    if not successful:
        raise ValueError("arm has no successful measured candidate")
    floor = float(ap70_ref) - float(max_ap_drop)
    feasible = [row for row in successful if row["ap70"] >= floor]
    pool = feasible or successful
    minimum_latency = min(row["latency_ms"] for row in pool)
    tied = [
        row
        for row in pool
        if row["latency_ms"] <= minimum_latency * 1.01
    ]
    selected = min(tied, key=lambda row: (row["energy_j"], row["latency_ms"], row["row_id"]))
    return {
        **selected,
        "ap70_floor": floor,
        "ap_constraint_satisfied": bool(feasible),
        "selection_status": (
            "selected_feasible"
            if feasible
            else "selected_fastest_ap_constraint_violation"
        ),
    }


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _feedback_rows(root: Path) -> list[dict[str, Any]]:
    return [_read(path) for path in sorted(root.glob("execution/*/feedback_row.json"))]


def _median_performance(
    paths: Sequence[Path],
    *,
    expected_onnx_sha256: str,
    expected_precision: str,
    expected_builder_level: int,
    expected_gpu: int,
) -> dict[str, Any]:
    if len(paths) != 3:
        raise ValueError(f"expected three independent performance repeats, got {len(paths)}")
    reports = [_read(path) for path in paths]
    for report in reports:
        if report.get("precision") != expected_precision:
            raise ValueError("independent repeat precision drift")
        if int(report.get("builder_optimization_level", -1)) != expected_builder_level:
            raise ValueError("independent repeat builder level drift")
        if int(report.get("gpu_abs", -1)) != expected_gpu:
            raise ValueError("independent repeat GPU drift")
        artifacts = report.get("artifact_sha256") or {}
        if artifacts.get("source_onnx") != expected_onnx_sha256:
            raise ValueError("independent repeat ONNX SHA drift")
        if not isinstance(artifacts.get("compiled_engine"), str):
            raise ValueError("independent repeat engine SHA missing")
    return {
        "latency_ms": float(statistics.median(float(row["lat_p50_ms"]) for row in reports)),
        "energy_j": float(statistics.median(float(row["energy_j"]) for row in reports)),
        "repeat_paths": [str(path.resolve()) for path in paths],
        "repeat_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
        "repeat_engine_sha256": [
            str(report["artifact_sha256"]["compiled_engine"]) for report in reports
        ],
    }


def _median_native(
    paths: Sequence[Path],
    *,
    expected_checkpoint_sha256: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    if len(paths) != 3:
        raise ValueError(f"expected three independent native repeats, got {len(paths)}")
    reports = [_read(path) for path in paths]
    for report in reports:
        if report.get("checkpoint_sha256") != expected_checkpoint_sha256:
            raise ValueError("native repeat checkpoint SHA drift")
        if report.get("config_sha256") != expected_config_sha256:
            raise ValueError("native repeat config SHA drift")
    return {
        "latency_ms": float(statistics.median(float(row["latency_ms"]) for row in reports)),
        "energy_j": float(statistics.median(float(row["energy_j"]) for row in reports)),
        "repeat_paths": [str(path.resolve()) for path in paths],
        "repeat_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
    }


def _configuration(row: Mapping[str, Any]) -> str:
    return f"({','.join(map(str, row['width']))},{row['q_mode']})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_path = root / "contracts/frozen_contract.json"
    contract = _read(contract_path)
    ap70_ref = float(contract["ap70_ref"])
    max_drop = 0.10

    native_metrics = _median_native(
        sorted((root / "controls/original_default").glob("same_gpu7_repeat_*.json")),
        expected_checkpoint_sha256=contract["checkpoint_sha256"],
        expected_config_sha256=contract["config_sha256"],
    )
    original = {
        "row_id": "fcooper-original-default",
        "terminal_status": SUCCESS,
        "width": [64, 128, 256, 128, 256],
        "q_mode": "fp32",
        "ap30": float(contract["ap30_ref"]),
        "ap50": float(contract["ap50_ref"]),
        "ap70": ap70_ref,
        "ap_report_sha256": contract["ap_reference_report_sha256"],
        "checkpoint_sha256": contract["checkpoint_sha256"],
        **native_metrics,
        "ap_constraint_satisfied": True,
        "selection_status": "fixed_original_default",
    }

    schedule_rows = _feedback_rows(root / "controls/schedule_only")
    if len(schedule_rows) != 1:
        raise ValueError("schedule-only requires exactly one full-AP feedback row")
    schedule_metrics = _median_performance(
        sorted(
            (root / "controls/schedule_only").glob(
                "same_gpu7_repeat_*/performance.json"
            )
        ),
        expected_onnx_sha256=schedule_rows[0]["graph_features"]["onnx_sha256"],
        expected_precision="fp32",
        expected_builder_level=5,
        expected_gpu=7,
    )
    schedule = {
        **schedule_rows[0],
        **schedule_metrics,
        "ap_constraint_satisfied": schedule_rows[0]["ap70"] >= ap70_ref - max_drop,
        "selection_status": "fixed_schedule_only",
    }

    compression_initial = _read(
        root / "controls/compression_only/selected_initial.json"
    )
    compression_metrics = _median_performance(
        sorted(
            (root / "controls/compression_only").glob(
                "selected_same_gpu7_repeat_*/performance.json"
            )
        ),
        expected_onnx_sha256=compression_initial["graph_features"]["onnx_sha256"],
        expected_precision=str(compression_initial["q_mode"]),
        expected_builder_level=0,
        expected_gpu=7,
    )
    compression = {**compression_initial, **compression_metrics}

    ctt_initial = _read(root / "controls/compress_then_tune/selected_initial.json")
    ctt_metrics = _median_performance(
        sorted(
            (root / "controls/compress_then_tune").glob(
                "selected_same_gpu7_repeat_*/performance.json"
            )
        ),
        expected_onnx_sha256=ctt_initial["graph_features"]["onnx_sha256"],
        expected_precision=str(ctt_initial["q_mode"]),
        expected_builder_level=5,
        expected_gpu=7,
    )
    compress_then_tune = {**ctt_initial, **ctt_metrics}

    incumbents = _feedback_rows(root / "probes/formal_incumbents")
    online_rows = _read(
        root / "search/S5-FCO-TRT/feedback_history_final_t16.json"
    )["rows"]
    gear_initial = select_arm_candidate(
        [*incumbents, *online_rows],
        ap70_ref=ap70_ref,
        max_ap_drop=max_drop,
    )
    if gear_initial["row_id"] not in {row["row_id"] for row in incumbents}:
        raise ValueError("GEAR selected an online row without bound independent repeats")
    gear_metrics = _median_performance(
        sorted(
            (root / "probes/formal_incumbents").glob(
                "int8_same_gpu7_repeat_*/performance.json"
            )
        ),
        expected_onnx_sha256=gear_initial["graph_features"]["onnx_sha256"],
        expected_precision=str(gear_initial["q_mode"]),
        expected_builder_level=5,
        expected_gpu=7,
    )
    gear = {
        **gear_initial,
        **gear_metrics,
        "selection_source": (
            "initial_probe_incumbent"
            if gear_initial["row_id"] in {row["row_id"] for row in incumbents}
            else "t16_online_feedback"
        ),
    }

    arm_rows = [
        ("Original/default", original, 1),
        ("Compression only", compression, 16),
        ("Schedule only", schedule, 1),
        ("Compress -> Tune", compress_then_tune, "12+4"),
        ("GEAR (Ours)", gear, 16),
    ]
    csv_rows = []
    for method, row, budget in arm_rows:
        csv_rows.append(
            {
                "method": method,
                "terminal_status": row["terminal_status"],
                "selection_status": row["selection_status"],
                "ap_constraint_satisfied": bool(row["ap_constraint_satisfied"]),
                "configuration": _configuration(row),
                "ap30": float(row["ap30"]),
                "ap50": float(row["ap50"]),
                "ap70": float(row["ap70"]),
                "latency_ms": float(row["latency_ms"]),
                "energy_j": float(row["energy_j"]),
                "budget": budget,
                "failure_rate": 0.0,
            }
        )
    csv_path = output / "fcooper_stage6_trt_delta_ap_0.10.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    integrity_paths = [
        root / "search/S5-FCO-TRT/online_file_integrity_audit.json",
        root / "controls/compression_only/file_integrity_audit.json",
        root / "controls/compress_then_tune/file_integrity_audit.json",
        root / "probes/formal_incumbents/file_integrity_audit.json",
        root / "search/S5-FCO-TRT/final_feedback_integrity_audit.json",
    ]
    integrity = [_read(path) for path in integrity_paths]
    integrity_passed = all(payload.get("passed") is True for payload in integrity)
    audit = {
        "schema_version": "stage6_fcooper_five_arm_audit_v1",
        "target_model": "fcooper",
        "hardware_id": "h800",
        "backend": "trt",
        "performance_gpu_index": 7,
        "ap70_ref": ap70_ref,
        "ap70_floor": ap70_ref - max_drop,
        "model_test_samples": int(contract["dataset_samples"]),
        "online_search_budget_consumed": len(online_rows),
        "online_search_max_ap70": max(float(row["ap70"]) for row in online_rows),
        "gear_selection_source": gear["selection_source"],
        "strict_feasible_arms": [
            row["method"] for row in csv_rows if row["ap_constraint_satisfied"]
        ],
        "constraint_violation_arms": [
            row["method"] for row in csv_rows if not row["ap_constraint_satisfied"]
        ],
        "integrity_audits": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "passed": payload.get("passed"),
                "row_count": payload.get("row_count"),
            }
            for path, payload in zip(integrity_paths, integrity)
        ],
        "paper_ready": integrity_passed
        and len(csv_rows) == 5
        and gear["ap_constraint_satisfied"]
        and all(len(row["repeat_paths"]) == 3 for _, row, _ in arm_rows),
        "rows": csv_rows,
        "selected_evidence": {
            method: {
                "row_id": row["row_id"],
                "engine_sha256": row.get("engine_sha256"),
                "performance_result_sha256": row.get("performance_result_sha256"),
                "ap_report_sha256": row.get("ap_report_sha256"),
                "checkpoint_sha256": row.get("checkpoint_sha256"),
                "materialized_source_evidence_sha256": row.get(
                    "materialized_source_evidence_sha256"
                ),
                "repeat_paths": row["repeat_paths"],
                "repeat_sha256": row["repeat_sha256"],
                "repeat_engine_sha256": row.get("repeat_engine_sha256"),
            }
            for method, row, _ in arm_rows
        },
    }
    audit_path = output / "fcooper_stage6_five_arm_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    bundle = {
        "schema_version": "fcooper_stage6_evidence_bundle_v1",
        "audit_path": str(audit_path),
        "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "csv_path": str(csv_path),
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "frozen_contract_path": str(contract_path),
        "frozen_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "selected_evidence": audit["selected_evidence"],
    }
    bundle_path = output / "fcooper_stage6_evidence_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"paper_ready": audit["paper_ready"], "rows": csv_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
