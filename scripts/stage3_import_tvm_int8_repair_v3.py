#!/usr/bin/env python3
"""Import repaired TVM INT8 build/AP evidence into Stage3 state JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def valid_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def evidence_paths(root: Path, model: str, width: str) -> tuple[Path, Path]:
    job = root / model / width
    percentile_ap = job / "ap_full_percentile_99_99" / "full_ap_eval_report.json"
    if percentile_ap.is_file():
        label = f"{model}_{width}_percentile_99_99"
        return job / "build_percentile_99_99" / label / "route_b_int8_auto_decomp_result.json", percentile_ap
    final_build = job / "build" / f"{model}_{width}_scaleaware_final" / "route_b_int8_auto_decomp_result.json"
    if final_build.is_file():
        return final_build, job / "ap_full" / "full_ap_eval_report.json"
    label = f"{model}_{width}_scaleaware"
    return job / "build" / label / "route_b_int8_auto_decomp_result.json", job / "ap_full" / "full_ap_eval_report.json"


def validate_evidence(build_path: Path, ap_path: Path, *, model: str) -> None:
    if not build_path.is_file() or not ap_path.is_file():
        raise ValueError(f"missing repaired evidence: {build_path} / {ap_path}")
    build = read_json(build_path)
    if build.get("correctness_all_exact") is not True:
        raise ValueError(f"candidate/native exact gate failed: {build_path}")
    latency_payload = build.get("latency") if isinstance(build.get("latency"), dict) else {}
    energy_payload = build.get("energy") if isinstance(build.get("energy"), dict) else {}
    latency = build.get("lat_p50_ms", build.get("latency_ms", latency_payload.get("latency_ms_p50")))
    energy = build.get("energy_j", energy_payload.get("energy_J", energy_payload.get("joule_per_inference")))
    if not valid_number(latency) or not valid_number(energy):
        raise ValueError(f"latency/energy missing: {build_path}")
    ap = read_json(ap_path)
    if ap.get("status") != "success" or ap.get("processed_samples") != 1789:
        raise ValueError(f"full AP report incomplete: {ap_path}")
    if ap.get("fallback_samples", 0) != 0 or ap.get("failed_samples", 0) != 0:
        raise ValueError(f"fallback/failed samples present: {ap_path}")
    if not all(valid_number(ap.get(key, ap.get("ap", {}).get(key))) for key in ("ap30", "ap50", "ap70")):
        raise ValueError(f"AP values missing: {ap_path}")
    gates = ap.get("gates") if isinstance(ap.get("gates"), dict) else {}
    if model == "codriving":
        gold_gate_passed = gates.get("full_1789") is True
    elif model == "pyramid":
        legacy_gate = ap.get("smoke_gate_passed") is True
        numeric_gate = (
            gates.get("full_1789") is True
            and ap.get("ap_row_allowed") is True
            and ap.get("feasibility_blockers") in (None, [])
        )
        gold_gate_passed = ap.get("ap_measured") is True and (legacy_gate or numeric_gate)
    else:
        raise ValueError(f"unsupported model for repaired evidence: {model}")
    if not gold_gate_passed:
        raise ValueError(f"final Gold gate failed: {ap_path}")


def import_rows(
    plan_rows: list[dict[str, Any]], root: Path, *, expected_rows: int = 24
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    performance: list[dict[str, Any]] = []
    ap_state: list[dict[str, Any]] = []
    selected = [row for row in plan_rows if row.get("q") == "int8" and "h800-tvm" in str(row.get("profile"))]
    if len(selected) != expected_rows:
        raise ValueError(f"expected {expected_rows} TVM INT8 rows, got {len(selected)}")
    for row in selected:
        model = str(row["model"])
        width = "x".join(str(value) for value in row["width"])
        build_path, ap_path = evidence_paths(root, model, width)
        validate_evidence(build_path, ap_path, model=model)
        performance.append({
            "job_id": str(row["performance_job_id"]), "manifest_job_id": str(row["performance_job_id"]),
            "status": "success", "result_json": str(build_path.resolve()),
            "source": "stage3_tvm_int8_repair_v3",
        })
        ap_payload = read_json(ap_path)
        ap_state.append({
            "schema_version": "stage3_gold96_ap_seed_state_v3",
            "record_type": "job_terminal", "manifest_job_id": str(row["manifest_job_id"]),
            "job_id": str(row["manifest_job_id"]), "stage": "full", "status": "success",
            "report_path": str(ap_path.resolve()),
            "report_sha256": hashlib.sha256(ap_path.read_bytes()).hexdigest(),
            "ap": {
                key: float(ap_payload.get(key, ap_payload.get("ap", {}).get(key)))
                for key in ("ap30", "ap50", "ap70")
            },
            "source": "stage3_tvm_int8_repair_v3",
        })
    return performance, ap_state


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ap-plan-jsonl", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--performance-state-output", type=Path, required=True)
    parser.add_argument("--ap-state-output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=24)
    args = parser.parse_args()
    plan = [json.loads(line) for line in args.ap_plan_jsonl.read_text().splitlines() if line.strip()]
    performance, ap_state = import_rows(plan, args.repair_root, expected_rows=args.expected_rows)
    write_jsonl(args.performance_state_output, performance)
    write_jsonl(args.ap_state_output, ap_state)
    print(json.dumps({"performance_rows": len(performance), "ap_rows": len(ap_state)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
