#!/usr/bin/env python3
"""Generate and audit the Stage6 six-arm launch contract without starting Stage6."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage6.contracts_v1 import (  # noqa: E402
    ARM_IDS,
    build_stage6_manifest,
    validate_stage6_manifest,
)


EXPECTED_BASELINE = {
    "schema_version": "stage6_native_fp32_baseline_v1",
    "hardware": "NVIDIA H800",
    "input_shape": [2, 64, 128, 256],
    "scope": "pyramid_multiscale_backbone",
    "precision": "fp32",
    "backend": "pytorch_eager",
    "width": [64, 128, 256],
    "backend_tuning": False,
    "full_network_claim": False,
}


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_path(value: Any, *, evidence_path: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else evidence_path.parent / path


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def audit_joint_closure(path: Path, *, require_artifacts: bool = False) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "passed": False, "failures": ["missing"]}
    row = _read(path)
    checks = {
        "closure": row.get("closure") is True,
        "sample_budget_16": row.get("sample_budget") == 16,
        "batch_size_4": row.get("batch_size") == 4,
        "round_count_4": row.get("round_count") == 4,
        "online_count_16": row.get("online_count") == 16,
    }
    artifact_audits = []
    if require_artifacts:
        for point in row.get("frontier_points") or []:
            for path_key, sha_key in (
                ("performance_result_json", "performance_result_sha256"),
                ("ap_report_path", "ap_report_sha256"),
            ):
                artifact = Path(str(point.get(path_key) or ""))
                expected = str(point.get(sha_key) or "")
                passed = artifact.is_file() and bool(expected) and _sha(artifact) == expected
                artifact_audits.append(
                    {
                        "row_id": point.get("manifest_job_id"),
                        "kind": path_key,
                        "path": str(artifact),
                        "passed": passed,
                    }
                )
        checks["frontier_artifacts_sha_bound"] = bool(artifact_audits) and all(
            item["passed"] for item in artifact_audits
        )
    return {
        "path": str(path),
        "sha256": _sha(path),
        "task_id": row.get("task_id"),
        "checks": checks,
        "artifact_audits": artifact_audits,
        "passed": all(checks.values()),
        "failures": [name for name, passed in checks.items() if not passed],
    }


def classify_baseline_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "classification": "invalid_scope", "mismatches": ["missing"]}
    row = _read(path)
    observed = {
        "schema_version": row.get("schema_version"),
        "hardware": row.get("hardware") or row.get("gpu"),
        "input_shape": row.get("input_shape"),
        "scope": row.get("scope"),
        "precision": row.get("precision"),
        "backend": row.get("backend"),
        "width": row.get("width"),
        "backend_tuning": row.get("backend_tuning"),
        "full_network_claim": row.get("full_network_claim"),
    }
    mismatches = [key for key, expected in EXPECTED_BASELINE.items() if observed.get(key) != expected]
    if float(row.get("latency_p50_ms") or 0) <= 0:
        mismatches.append("latency_p50_ms")
    if float(row.get("energy_j") or 0) <= 0:
        mismatches.append("energy_j")
    if len(str(row.get("checkpoint_sha256") or "")) != 64:
        mismatches.append("checkpoint_sha256")
    if not row.get("output_shapes"):
        mismatches.append("output_shapes")
    has_repeats = int(row.get("independent_repeat_count") or 0) >= 3
    if mismatches:
        classification = "invalid_scope"
    elif has_repeats:
        classification = "reusable"
    else:
        classification = "repeat_only"

    paper_table_mismatches = []
    ap70 = row.get("ap70")
    if (
        isinstance(ap70, bool)
        or not isinstance(ap70, (int, float))
        or not math.isfinite(float(ap70))
        or float(ap70) < 0
    ):
        paper_table_mismatches.append("ap70")
    ap_report = _artifact_path(row.get("ap_report_path"), evidence_path=path)
    if ap_report is None or not ap_report.is_file():
        paper_table_mismatches.append("ap_report_path")
    expected_ap_sha = str(row.get("ap_report_sha256") or "")
    if (
        len(expected_ap_sha) != 64
        or ap_report is None
        or not ap_report.is_file()
        or _sha(ap_report) != expected_ap_sha
    ):
        paper_table_mismatches.append("ap_report_sha256")
    return {
        "path": str(path),
        "sha256": _sha(path),
        "classification": classification,
        "mismatches": mismatches,
        "paper_table_eligible": classification == "reusable" and not paper_table_mismatches,
        "paper_table_mismatches": paper_table_mismatches,
        "observed": observed,
    }


def audit_formal_runner_plan(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "passed": False,
            "failures": ["formal_runner_plan_argument_missing"],
        }
    if not path.is_file():
        return {"path": str(path), "passed": False, "failures": ["missing"]}

    row = _read(path)
    runner = _artifact_path(row.get("formal_runner_path"), evidence_path=path)
    expected_runner_sha = str(row.get("formal_runner_sha256") or "")
    checks = {
        "schema_version": row.get("schema_version") == "stage6_formal_runner_plan_v1",
        "experiment_id": row.get("experiment_id") == "stage6-pyramid-h800-six-arm-v1",
        "target_model": row.get("target_model") == "pyramid",
        "hardware_id": row.get("hardware_id") == "h800",
        "backends": row.get("backends") == ["tvm", "trt"],
        "six_arm_coverage": row.get("arm_ids") == list(ARM_IDS),
        "formal_runner_exists": runner is not None and runner.is_file(),
        "formal_runner_sha_bound": (
            runner is not None
            and runner.is_file()
            and len(expected_runner_sha) == 64
            and _sha(runner) == expected_runner_sha
        ),
        "output_root_declared": bool(row.get("output_root")),
        "budget_contract_bound": row.get("budget_contract_bound") is True,
    }
    return {
        "path": str(path),
        "sha256": _sha(path),
        "formal_runner_path": str(runner) if runner is not None else None,
        "checks": checks,
        "passed": all(checks.values()),
        "failures": [name for name, passed in checks.items() if not passed],
    }


def audit_joint_actual_feedback(summary_path: Path, rows_path: Path) -> dict[str, Any]:
    if not summary_path.is_file() or not rows_path.is_file():
        return {"passed": False, "failures": ["missing_summary_or_online_rows"]}
    summary = _read(summary_path)
    with rows_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row_audits = []
    for row in rows:
        graph_features = json.loads(row.get("graph_features") or "{}")
        actual_features = (
            row.get("feedback_feature_contract") == "actual_feedback_v3"
            and graph_features.get("graph_feature_provenance") == "materialized_onnx_extracted_v1"
            and len(row.get("materialized_graph_features_sha256") or "") == 64
        )
        terminal = row.get("terminal_status")
        if terminal == "measured_success_gold":
            evidence_pairs = (
                (row.get("performance_result_json"), row.get("performance_result_sha256")),
                (row.get("ap_report_path"), row.get("ap_report_sha256")),
            )
        else:
            evidence_pairs = ((row.get("failure_evidence_path"), row.get("failure_evidence_sha256")),)
        evidence_bound = all(
            bool(path_value)
            and Path(str(path_value)).is_file()
            and bool(expected)
            and _sha(Path(str(path_value))) == expected
            for path_value, expected in evidence_pairs
        )
        row_audits.append(
            {
                "row_id": row.get("manifest_job_id"),
                "terminal_status": terminal,
                "actual_features": actual_features,
                "evidence_sha_bound": evidence_bound,
                "passed": actual_features and evidence_bound,
            }
        )
    checks = {
        "summary_status_closed": summary.get("status") == "pyramid_search_closed",
        "summary_online_rows_32": summary.get("online_row_count") == 32,
        "summary_actual_features_32": summary.get("actual_graph_feature_count") == 32,
        "summary_no_silent_fallback": summary.get("silent_surrogate_fallback_count") == 0,
        "online_csv_rows_32": len(rows) == 32,
        "all_online_rows_actual_and_sha_bound": bool(row_audits)
        and all(row["passed"] for row in row_audits),
    }
    return {
        "summary_path": str(summary_path),
        "summary_sha256": _sha(summary_path),
        "online_rows_path": str(rows_path),
        "online_rows_sha256": _sha(rows_path),
        "checks": checks,
        "row_audits": row_audits,
        "passed": all(checks.values()),
        "failures": [name for name, passed in checks.items() if not passed],
    }


def build_gate(
    output_dir: Path,
    baseline_candidates: list[Path],
    *,
    formal_runner_plan: Path | None = None,
) -> dict[str, Any]:
    manifest = build_stage6_manifest()
    manifest_audit = validate_stage6_manifest(manifest)
    joint = {
        backend: audit_joint_closure(ROOT / relative, require_artifacts=True)
        for backend, relative in manifest["arms"][-1]["joint_evidence"].items()
    }
    joint_arm = manifest["arms"][-1]
    joint_actual = audit_joint_actual_feedback(
        ROOT / joint_arm["joint_summary_evidence"],
        ROOT / joint_arm["joint_online_rows_evidence"],
    )
    baselines = [classify_baseline_evidence(path) for path in baseline_candidates]
    reusable_baseline = next(
        (row for row in baselines if row["classification"] == "reusable"), None
    )
    runner_smoke = output_dir / "stage6_runner_smoke_v1.json"
    smoke = _read(runner_smoke) if runner_smoke.is_file() else {
        "passed": False,
        "failures": ["runner_smoke_not_executed"],
    }
    protocol_gates = {
        "manifest_contract": manifest_audit["passed"],
        "joint_evidence_tvm": joint["tvm"]["passed"],
        "joint_evidence_trt": joint["trt"]["passed"],
        "joint_actual_feedback_32": joint_actual["passed"],
        "native_fp32_baseline": reusable_baseline is not None,
        "runner_smoke": smoke.get("passed") is True,
    }
    protocol_ready = all(protocol_gates.values())
    formal_plan_audit = audit_formal_runner_plan(formal_runner_plan)
    formal_execution_gates = {
        "protocol_ready": protocol_ready,
        "formal_runner_plan": formal_plan_audit["passed"],
    }
    formal_execution_ready = all(formal_execution_gates.values())
    paper_baseline = next(
        (row for row in baselines if row.get("paper_table_eligible") is True), None
    )
    paper_table_gates = {
        "formal_execution_ready": formal_execution_ready,
        "native_fp32_full_ap": paper_baseline is not None,
        "formal_stage6_results_complete": False,
    }
    paper_table_ready = all(paper_table_gates.values())
    launch_allowed = formal_execution_ready
    manifest = {
        **manifest,
        "protocol_ready": protocol_ready,
        "formal_execution_ready": formal_execution_ready,
        "paper_table_ready": paper_table_ready,
        "paper_eligible": paper_table_ready,
        "launch_allowed": launch_allowed,
        "paper_eligibility_reason": (
            "formal Stage6 rows and independent validation are not yet complete"
            if not paper_table_ready
            else None
        ),
        "launch_gate_report": "stage6_launch_gate_report_v1.json",
    }
    report = {
        "schema_version": "stage6_launch_gate_report_v1",
        "protocol_ready": protocol_ready,
        "formal_execution_ready": formal_execution_ready,
        "paper_table_ready": paper_table_ready,
        "launch_allowed": launch_allowed,
        "gates": protocol_gates,
        "protocol_gates": protocol_gates,
        "formal_execution_gates": formal_execution_gates,
        "paper_table_gates": paper_table_gates,
        "manifest_audit": manifest_audit,
        "joint_evidence": joint,
        "joint_actual_feedback": joint_actual,
        "baseline_evidence_audit": baselines,
        "bound_native_fp32_baseline": reusable_baseline,
        "bound_native_fp32_paper_baseline": paper_baseline,
        "runner_smoke": smoke,
        "formal_runner_plan_audit": formal_plan_audit,
        "formal_stage6_started": False,
    }
    _write(output_dir / "stage6_pyramid_arm_manifest_v2.json", manifest)
    _write(output_dir / "stage6_launch_gate_report_v1.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/stage6_pyramid_launch_gate_20260720",
    )
    parser.add_argument("--baseline-candidate", type=Path, action="append", default=[])
    parser.add_argument(
        "--formal-runner-plan",
        type=Path,
        help="Machine-readable Stage6 formal runner plan; required for launch_allowed=true.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_gate(
        args.output_dir,
        list(args.baseline_candidate),
        formal_runner_plan=args.formal_runner_plan,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["launch_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
