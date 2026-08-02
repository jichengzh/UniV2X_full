#!/usr/bin/env python3
"""Finalize independent schedule-only repeats for TVM and TensorRT."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.stage5_finalize_independent_validation_v1 import evaluate_consistency  # noqa: E402


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ap70(report: Mapping[str, Any]) -> float:
    source = report.get("ap") if isinstance(report.get("ap"), Mapping) else report
    return float(source["ap70"])


def _repeat(path: Path, backend: str, index: int) -> dict[str, Any]:
    payload = _read(path)
    latency_key = "lat_tuned_ms" if backend == "tvm" else "lat_p50_ms"
    if payload.get("build_success") is not True:
        raise ValueError(f"schedule-only repeat failed: {path}")
    return {
        "status": "success",
        "repeat_id": f"repeat-{index}",
        "latency_ms": float(payload[latency_key]),
        "energy_j": float(payload["energy_j"]),
        "performance_result_json": str(path),
        "performance_result_sha256": _sha(path),
    }


def _task(
    root: Path, bundle: Mapping[str, Any], backend: str, target_model: str
) -> dict[str, Any]:
    arm = bundle["backends"][backend]["schedule_only"]
    if arm.get("status") != "complete" or len(arm.get("points") or []) != 1:
        raise ValueError(f"schedule-only search evidence is incomplete: {backend}")
    reference = arm["points"][0]
    backend_root = root / backend
    repeats = [
        _repeat(backend_root / f"repeat_{index}/performance_result.json", backend, index)
        for index in range(3)
    ]
    ap_path = backend_root / "ap/full_1789/full_ap_eval_report.json"
    ap_payload = _read(ap_path)
    common_ap_invalid = (
        int(ap_payload.get("processed_samples") or 0) != 1789
        or int(ap_payload.get("failed_samples") or 0) != 0
        or int(ap_payload.get("fallback_samples") or 0) != 0
    )
    pyramid_tvm_invalid = target_model == "pyramid" and backend == "tvm" and (
        ap_payload.get("smoke_gate_passed") is not True
        or int(ap_payload.get("pred_nonempty_count") or 0) <= 0
    )
    codriving_invalid = target_model == "codriving" and (
        ap_payload.get("status") != "success"
        or ap_payload.get("schema")
        not in {
            "stage3_codriving_tvm_fp16_ap_bridge_v3",
            "stage3_codriving_trt_multiscale_ap_bridge_v3",
        }
        or int(ap_payload.get("engine_samples") or 0) != 1789
    )
    if common_ap_invalid or pyramid_tvm_invalid or codriving_invalid:
        raise ValueError(f"schedule-only independent AP contract mismatch: {backend}")
    ap = {"ap70": _ap70(ap_payload)}
    consistency = evaluate_consistency(
        {
            "latency_ms": float(reference["latency_ms"]),
            "energy_j": float(reference["energy_j"]),
            "ap70": float(reference["AP70"]),
        },
        repeats,
        ap,
        raise_on_failure=False,
    )
    evidence_path = (
        backend_root / "repeat_2/artifact_report.json"
        if backend == "tvm"
        else backend_root / "repeat_2/artifacts/engine_build_config.json"
    )
    configuration = {
        "arm_id": "schedule_only",
        "pipeline_id": f"{backend}:schedule_only:base_fp32:tuning_trials=64",
        "configuration_id": str(reference["manifest_job_id"]),
        "consistency": consistency,
        "performance_repeats": [
            {
                "repeat_id": repeat["repeat_id"],
                "performance_result_json": repeat["performance_result_json"],
                "performance_result_sha256": repeat["performance_result_sha256"],
            }
            for repeat in repeats
        ],
        "ap_report_path": str(ap_path),
        "ap_report_sha256": _sha(ap_path),
        "evidence_path": str(evidence_path),
        "evidence_sha256": _sha(evidence_path),
    }
    return {
        "task_id": f"{backend}:schedule_only",
        "arm_id": "schedule_only",
        "backend": backend,
        "passed": consistency["passed"],
        "configurations": [configuration],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evidence-bundle", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--target-model", choices=("pyramid", "codriving"), default="pyramid")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = _read(args.evidence_bundle)
    tasks = [
        _task(args.root, bundle, backend, args.target_model)
        for backend in ("tvm", "trt")
    ]
    audit = {
        "schema_version": "stage6_independent_validation_audit_v1",
        "all_tasks_passed": all(task["passed"] for task in tasks),
        "task_count": len(tasks),
        "configuration_count": 2,
        "tasks": tasks,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_json": str(args.output_json), "all_tasks_passed": audit["all_tasks_passed"]}, sort_keys=True))
    return 0 if audit["all_tasks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
