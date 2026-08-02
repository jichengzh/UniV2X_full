#!/usr/bin/env python3
"""Convert the Pyramid TRT schedule repair into a standard Stage6 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    repair_path = args.repair_root / "repair_audit.json"
    repair = _read(repair_path)
    if (
        repair.get("status") != "passed"
        or repair.get("checkpoint_epoch") != 23
        or repair.get("processed_samples") != 1789
    ):
        raise ValueError("Pyramid TRT schedule repair is not passed")

    ap_path = Path(repair["ap_report_path"])
    if _sha(ap_path) != repair["ap_report_sha256"]:
        raise ValueError("repair AP SHA mismatch")
    repeats = repair.get("performance_repeats") or []
    if len(repeats) != 3:
        raise ValueError("repair requires three performance repeats")
    performance_repeats = []
    for index, repeat in enumerate(repeats):
        path = Path(repeat["path"])
        if _sha(path) != repeat["sha256"]:
            raise ValueError(f"repair performance SHA mismatch: repeat {index}")
        performance_repeats.append(
            {
                "repeat_id": f"repeat-{index}",
                "performance_result_json": str(path),
                "performance_result_sha256": repeat["sha256"],
            }
        )

    evidence_path = args.repair_root / "performance/repeat_2/artifacts/engine_build_config.json"
    reference = {
        "ap70": 0.6311015785876983,
        "latency_ms": 1.0965440273284912,
        "energy_j": 0.4706979215523819,
    }
    rerun = {
        "ap70": float(repair["ap70"]),
        "latency_median_ms": float(repair["latency_median_ms"]),
        "energy_median_j": float(repair["energy_median_j"]),
        "latency_cv": float(repair["latency_cv"]),
        "energy_cv": float(repair["energy_cv"]),
    }
    deltas = {
        "ap70_absolute": abs(rerun["ap70"] - reference["ap70"]),
        "latency_relative": abs(rerun["latency_median_ms"] - reference["latency_ms"])
        / reference["latency_ms"],
        "energy_relative": abs(rerun["energy_median_j"] - reference["energy_j"])
        / reference["energy_j"],
    }
    thresholds = {
        "ap70_absolute": 0.01,
        "latency_relative": 0.15,
        "energy_relative": 0.20,
        "latency_cv": 0.10,
        "energy_cv": 0.15,
    }
    passed = all(
        (
            deltas["ap70_absolute"] <= thresholds["ap70_absolute"],
            deltas["latency_relative"] <= thresholds["latency_relative"],
            deltas["energy_relative"] <= thresholds["energy_relative"],
            rerun["latency_cv"] <= thresholds["latency_cv"],
            rerun["energy_cv"] <= thresholds["energy_cv"],
        )
    )
    configuration = {
        "configuration_id": "stage6|pyramid|schedule_only|trt|fp32",
        "arm_id": "schedule_only",
        "pipeline_id": "trt:schedule_only:base_fp32:tuning_trials=64:epoch23_repair",
        "performance_repeats": performance_repeats,
        "ap_report_path": str(ap_path),
        "ap_report_sha256": _sha(ap_path),
        "evidence_path": str(evidence_path),
        "evidence_sha256": _sha(evidence_path),
        "consistency": {
            "passed": passed,
            "reference": reference,
            "rerun": rerun,
            "deltas": deltas,
            "thresholds": thresholds,
        },
    }
    audit = {
        "schema_version": "stage6_independent_validation_audit_v1",
        "all_tasks_passed": passed,
        "task_count": 1,
        "configuration_count": 1,
        "tasks": [
            {
                "task_id": "trt:schedule_only:epoch23_repair",
                "arm_id": "schedule_only",
                "backend": "trt",
                "passed": passed,
                "configurations": [configuration],
            }
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "output_json": str(args.output_json)}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
