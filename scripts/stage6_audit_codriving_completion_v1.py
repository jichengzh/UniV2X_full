#!/usr/bin/env python3
"""Audit all frozen deliverables for the CoDriving Stage6 six-arm result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


ARMS = (
    "compression_only",
    "schedule_only",
    "compress_then_tune",
    "tune_then_compress",
    "joint_shcosearch",
)
ALLOWED_ORIGINS = {
    "stage6_new_original_default_measurement",
    "stage6_new_control_measurement",
    "joint_actual_v3_initial_observed_gold176",
    "joint_actual_v3_online_feedback",
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any) -> bool:
    try:
        return value not in (None, "", "-") and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _require_sha(path_value: Any, expected: Any, label: str) -> Path:
    path = Path(str(path_value or ""))
    if not path.is_file():
        raise ValueError(f"{label} file missing: {path}")
    if _sha(path) != str(expected or ""):
        raise ValueError(f"{label} SHA drift: {path}")
    return path


def _audit_independent_measurements(formal_root: Path) -> dict[str, Any]:
    audit_paths = [
        *sorted(
            (formal_root / "independent_validation_controls_parallel").glob(
                "stage6_independent_validation_audit_*.json"
            )
        ),
        formal_root
        / "independent_validation_joint/stage6_joint_independent_validation_audit_trt_v1.json",
        formal_root
        / "independent_validation_joint/stage6_joint_independent_validation_audit_tvm_v1.json",
        formal_root
        / "independent_validation_schedule/stage6_schedule_independent_validation_audit_v1.json",
    ]
    if len(audit_paths) != 7:
        raise ValueError(f"expected seven independent validation audits, got {len(audit_paths)}")

    configuration_count = 0
    performance_repeat_count = 0
    outputs = []
    for audit_path in audit_paths:
        audit = _read(audit_path)
        if (
            audit.get("schema_version") != "stage6_independent_validation_audit_v1"
            or audit.get("all_tasks_passed") is not True
        ):
            raise ValueError(f"independent validation audit is not passed: {audit_path}")
        observed_configurations = 0
        for task in audit.get("tasks") or []:
            if task.get("passed") is not True:
                raise ValueError(f"independent validation task failed: {audit_path}")
            for configuration in task.get("configurations") or []:
                observed_configurations += 1
                configuration_count += 1
                if configuration.get("consistency", {}).get("passed") is not True:
                    raise ValueError(
                        f"independent consistency failed: {configuration.get('configuration_id')}"
                    )

                ap_path = _require_sha(
                    configuration.get("ap_report_path"),
                    configuration.get("ap_report_sha256"),
                    "independent AP report",
                )
                ap = _read(ap_path)
                if not all(_finite(ap.get(metric)) for metric in ("ap30", "ap50", "ap70")):
                    raise ValueError(f"independent AP report misses AP30/AP50/AP70: {ap_path}")
                if (
                    int(ap.get("processed_samples") or 0) != 1789
                    or int(ap.get("failed_samples") or 0) != 0
                    or int(ap.get("fallback_samples") or 0) != 0
                ):
                    raise ValueError(f"independent AP report violates sample contract: {ap_path}")
                raw_ap_path = ap.get("raw_ap_report_path")
                raw_ap_sha = ap.get("raw_ap_report_sha256")
                if raw_ap_path or raw_ap_sha:
                    _require_sha(raw_ap_path, raw_ap_sha, "raw independent AP report")

                repeats = configuration.get("performance_repeats") or []
                if len(repeats) != 3:
                    raise ValueError(
                        f"expected three performance repeats: {configuration.get('configuration_id')}"
                    )
                for repeat in repeats:
                    performance_path = _require_sha(
                        repeat.get("performance_result_json"),
                        repeat.get("performance_result_sha256"),
                        "independent performance result",
                    )
                    performance = _read(performance_path)
                    performance_succeeded = performance.get("status") == "success" or (
                        performance.get("build_success") is True
                    )
                    latency = performance.get("latency_ms", performance.get("lat_p50_ms"))
                    if not performance_succeeded or not all(
                        _finite(value) for value in (latency, performance.get("energy_j"))
                    ):
                        raise ValueError(
                            f"independent performance result is incomplete: {performance_path}"
                        )
                    raw_performance_path = performance.get("raw_performance_result_json")
                    raw_performance_sha = performance.get("raw_performance_result_sha256")
                    if raw_performance_path or raw_performance_sha:
                        _require_sha(
                            raw_performance_path,
                            raw_performance_sha,
                            "raw independent performance result",
                        )
                    performance_repeat_count += 1

                _require_sha(
                    configuration.get("evidence_path"),
                    configuration.get("evidence_sha256"),
                    "independent artifact evidence",
                )
        if observed_configurations != int(audit.get("configuration_count") or 0):
            raise ValueError(f"independent configuration count drift: {audit_path}")
        outputs.append({"path": str(audit_path), "sha256": _sha(audit_path)})

    if configuration_count != 9 or performance_repeat_count != 27:
        raise ValueError(
            "independent measurement coverage drift: "
            f"configurations={configuration_count}, repeats={performance_repeat_count}"
        )
    return {
        "audit_count": len(outputs),
        "configuration_count": configuration_count,
        "performance_repeat_count": performance_repeat_count,
        "full_ap_sample_count_per_configuration": 1789,
        "audits": outputs,
    }


def _audit_native_baseline(formal_root: Path) -> dict[str, Any]:
    seal_path = formal_root / "stage6_native_fp32_baseline_sha_reseal_audit_v1.json"
    seal = _read(seal_path)
    if seal.get("status") != "passed" or seal.get("metrics_unchanged") is not True:
        raise ValueError("native FP32 baseline SHA reseal audit is not passed")
    baseline_path = _require_sha(
        seal.get("baseline_json"), seal.get("baseline_json_sha256"), "native baseline"
    )
    baseline = _read(baseline_path)
    if (
        baseline.get("schema_version") != "stage6_native_fp32_baseline_v3"
        or int(baseline.get("independent_repeat_count") or 0) != 3
        or len(baseline.get("latency_repeats") or []) != 3
        or int(baseline.get("ap_num_samples") or 0) != 1789
        or not all(
            _finite(baseline.get(metric))
            for metric in ("ap30", "ap50", "ap70", "latency_p50_ms", "energy_j")
        )
    ):
        raise ValueError("native FP32 baseline measurement contract is incomplete")
    _require_sha(
        baseline.get("ap_report_path"),
        baseline.get("ap_report_sha256"),
        "native baseline AP report",
    )
    return {
        "baseline_json": str(baseline_path),
        "baseline_json_sha256": _sha(baseline_path),
        "reseal_audit": str(seal_path),
        "reseal_audit_sha256": _sha(seal_path),
        "performance_repeat_count": 3,
        "full_ap_sample_count": 1789,
    }


def _audit_failure_evidence(bundle: Mapping[str, Any]) -> dict[str, Any]:
    outputs = {}
    total_attempts = 0
    for backend in ("tvm", "trt"):
        summary = bundle["backends"][backend]["tune_then_compress"]
        evidence = summary.get("attempt_evidence_sha256") or {}
        if (
            summary.get("status") != "complete_failure"
            or int(summary.get("planned_outer_genomes") or 0) != 16
            or int(summary.get("failure_count") or 0) != 16
            or len(evidence) != 16
            or summary.get("unmeasured_transfer_success_count") != 0
            or summary.get("points")
        ):
            raise ValueError(f"incomplete tune-then-compress failure coverage: {backend}")
        reasons = set()
        for path_value, expected_sha in evidence.items():
            path = _require_sha(path_value, expected_sha, "transfer failure evidence")
            attempt = _read(path)
            if (
                attempt.get("terminal_status") != "feasibility_failure"
                or attempt.get("fallback_used") is not False
                or attempt.get("full_transfer") is not False
                or attempt.get("compressed_shape_retuned") is not False
            ):
                raise ValueError(f"invalid transfer failure contract: {path}")
            reasons.add(str(attempt.get("failure_reason") or ""))
            total_attempts += 1
        outputs[backend] = {
            "attempt_count": len(evidence),
            "fallback_count": 0,
            "failure_reasons": sorted(reasons),
        }
    if total_attempts != 32:
        raise ValueError(f"expected 32 transfer failure attempts, got {total_attempts}")
    return {"total_attempt_count": total_attempts, "backends": outputs}


def _audit_requests(joint_root: Path) -> dict[str, Any]:
    closure = _read(joint_root / "closure/codriving_actual_v3_closure_audit.json")
    expected = {
        str(task["task_id"]): list(task["measurement_request_file_sha256"])
        for task in closure["tasks"]
    }
    actual = {}
    for task, shas in expected.items():
        paths = sorted((joint_root / task).glob("round_*/measurement_request.json"))
        observed = [_sha(path) for path in paths]
        if observed != shas:
            raise ValueError(f"frozen actual-v3 request SHA drift: {task}")
        actual[task] = {"request_count": len(paths), "request_sha256": observed}
    return actual


def _audit_formal_plan(formal_root: Path) -> dict[str, Any]:
    path = formal_root / "stage6_formal_execution_plan_v1.json"
    plan = _read(path)
    if plan.get("schema_version") != "stage6_formal_execution_plan_v1":
        raise ValueError("unexpected CoDriving formal execution plan schema")
    if plan.get("passed") is not True or plan.get("target_model") != "codriving":
        raise ValueError("CoDriving formal execution plan is not passed")
    if plan.get("hardware_blind_backend_labels_used") is not False:
        raise ValueError("formal ranking used backend labels")
    if plan.get("hardware_blind_cost_policy") != "parameter_bits_and_bitops":
        raise ValueError("formal ranking did not use the required bit-cost policy")
    if int(plan.get("effective_candidate_pool_size") or 0) != 648:
        raise ValueError("formal candidate pool size drift")

    plan_sha = str(plan.get("plan_sha256") or "")
    if not plan_sha:
        raise ValueError("formal execution plan contract SHA is missing")
    q_mode_counts: dict[str, int] = {}
    candidate_plan_count = 0
    for backend in ("tvm", "trt"):
        for arm in ("compression_only", "compress_then_tune", "tune_then_compress"):
            candidate_path = formal_root / backend / f"{arm}_candidate_plan.json"
            candidate = _read(candidate_path)
            if candidate.get("schema_version") != "stage6_formal_arm_candidate_plan_v1":
                raise ValueError(f"candidate plan schema drift: {backend}:{arm}")
            if candidate.get("backend") != backend or candidate.get("arm_id") != arm:
                raise ValueError(f"candidate plan identity drift: {backend}:{arm}")
            if candidate.get("plan_sha256") != plan_sha:
                raise ValueError(f"candidate plan contract SHA drift: {backend}:{arm}")
            expected = [tuple(genome) for genome in plan["backend_plans"][backend][arm]]
            observed = [
                (*map(int, row["width"]), str(row["q_mode"]))
                for row in candidate.get("rows") or []
            ]
            if observed != expected:
                raise ValueError(f"candidate plan genome drift: {backend}:{arm}")
            for genome in observed:
                q_mode_counts[genome[3]] = q_mode_counts.get(genome[3], 0) + 1
            candidate_plan_count += 1
    return {
        "path": str(path),
        "sha256": _sha(path),
        "plan_contract_sha256": plan_sha,
        "effective_candidate_pool_size": 648,
        "hardware_blind_backend_labels_used": False,
        "hardware_blind_cost_policy": "parameter_bits_and_bitops",
        "candidate_plan_count": candidate_plan_count,
        "selected_q_mode_counts": q_mode_counts,
    }


def _audit_tables(table_dir: Path, audit: Mapping[str, Any]) -> dict[str, Any]:
    if audit.get("paper_ready") is not True:
        raise ValueError("paper table audit is not paper_ready")
    outputs = {}
    for backend in ("tvm", "trt"):
        for delta in (0.05, 0.10):
            stem = f"codriving_stage6_{backend}_delta_ap_{delta:.2f}"
            csv_path = table_dir / f"{stem}.csv"
            md_path = table_dir / f"{stem}.md"
            if not csv_path.is_file() or not md_path.is_file():
                raise ValueError(f"paper table file missing: {stem}")
            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 6:
                raise ValueError(f"paper table must contain six arms: {stem}")
            methods = {row["method"] for row in rows}
            if methods != {"original_default", *ARMS}:
                raise ValueError(f"paper table arm identity drift: {stem}")
            for row in rows:
                origin = row.get("evidence_origin")
                if origin not in ALLOWED_ORIGINS:
                    raise ValueError(f"missing evidence origin: {stem}:{row['method']}")
                outcome = str(row.get("outcome") or "")
                if outcome.startswith("feasibility_failure"):
                    if any(_finite(row.get(metric)) for metric in ("AP70", "latency_ms", "energy_j")):
                        raise ValueError(f"failure row fabricates metrics: {stem}:{row['method']}")
                elif not all(_finite(row.get(metric)) for metric in ("AP70", "latency_ms", "energy_j")):
                    raise ValueError(f"measured row misses metrics: {stem}:{row['method']}")
            outputs[stem] = {
                "row_count": len(rows),
                "csv_sha256": _sha(csv_path),
                "markdown_sha256": _sha(md_path),
            }
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--joint-root", type=Path, required=True)
    parser.add_argument("--evidence-bundle", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    bundle = _read(args.evidence_bundle)
    if bundle.get("target_model") != "codriving":
        raise ValueError("evidence bundle target model mismatch")
    for backend in ("tvm", "trt"):
        summaries = bundle["backends"][backend]
        for arm in ARMS:
            summary = summaries[arm]
            status = summary.get("status")
            if status == "complete":
                if summary.get("independent_validation_complete") is not True:
                    raise ValueError(f"arm lacks independent validation: {backend}:{arm}")
            elif status == "complete_failure":
                if summary.get("failure_evidence_sha_verified") is not True:
                    raise ValueError(f"failure evidence is unverified: {backend}:{arm}")
            else:
                raise ValueError(f"arm is nonterminal: {backend}:{arm}:{status}")
    table_audit_path = args.table_dir / "stage6_codriving_paper_main_table_audit_v1.json"
    table_audit = _read(table_audit_path)
    result = {
        "schema_version": "stage6_codriving_completion_audit_v1",
        "status": "passed",
        "paper_ready": True,
        "target_model": "codriving",
        "formal_execution_plan": _audit_formal_plan(args.formal_root),
        "native_fp32_baseline": _audit_native_baseline(args.formal_root),
        "independent_measurements": _audit_independent_measurements(args.formal_root),
        "failure_measurements": _audit_failure_evidence(bundle),
        "frozen_actual_v3_requests": _audit_requests(args.joint_root),
        "tables": _audit_tables(args.table_dir, table_audit),
        "evidence_bundle": {
            "path": str(args.evidence_bundle),
            "sha256": _sha(args.evidence_bundle),
        },
        "paper_table_audit": {
            "path": str(table_audit_path),
            "sha256": _sha(table_audit_path),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
