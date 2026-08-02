#!/usr/bin/env python3
"""Collect SHA-audited Stage6 arm evidence into the paper-table schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from framework.stage6.evidence_bundle_v1 import (  # noqa: E402
    apply_independent_validation,
    attach_common_hypervolume,
    build_independent_validation_index,
    normalize_actual_feedback_row,
    normalize_gold_row,
)


BACKEND_TASK = {"tvm": "S5-PYR-TVM", "trt": "S5-PYR-TRT"}
EXPECTED_ONLINE = {
    "compression_only": 16,
    "compress_then_tune": 4,
    "joint_shcosearch": 16,
}
SUCCESS = "measured_success_gold"
FAILURES = {"feasibility_failure", "numerical_feasibility_failure"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(payload: Any) -> list[Mapping[str, Any]]:
    value = payload.get("rows") if isinstance(payload, Mapping) else payload
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("evidence rows must be a sequence")
    return [row for row in value if isinstance(row, Mapping)]


def _unique(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("manifest_job_id") or row.get("row_id") or "")
        if not row_id:
            raise ValueError("evidence row identity is missing")
        if row_id in result and result[row_id] != row:
            raise ValueError(f"conflicting duplicate evidence row: {row_id}")
        result[row_id] = row
    return list(result.values())


def _actual_arm(formal_root: Path, backend: str, arm: str) -> dict[str, Any]:
    task = BACKEND_TASK[backend]
    paths = sorted(
        (formal_root / backend / arm).glob(
            f"formal_batch_*/{task}/round_*/actual_feedback/stage5_feedback_v3_actual.json"
        )
    )
    raw = _unique([row for path in paths for row in _rows(_read(path))])
    points = [
        {
            **normalize_actual_feedback_row(row, expected_backend=backend),
            "evidence_origin": "stage6_new_control_measurement",
        }
        for row in raw
        if row.get("terminal_status") == SUCCESS
    ]
    failure_count = sum(row.get("terminal_status") in FAILURES for row in raw)
    expected = EXPECTED_ONLINE[arm]
    terminal_count = len(points) + failure_count
    status = "complete" if terminal_count == expected else "running"
    return {
        "status": status,
        "evidence_origin": "stage6_new_control_measurement",
        "independent_validation_complete": False,
        "points": points,
        "outer_genomes": terminal_count,
        "planned_outer_genomes": expected,
        "failure_count": failure_count,
        "tuning_trials": 0 if arm == "compression_only" else 64 * expected,
        "evidence_manifests": [str(path) for path in paths],
        "evidence_manifest_sha256": {str(path): _sha(path) for path in paths},
    }


def _joint_arm(
    *,
    backend: str,
    joint_root: Path,
    coldstart_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    task = BACKEND_TASK[backend]
    task_root = joint_root / task
    candidate_manifest = _read(task_root / "candidate_manifest.json")
    initial_ids = {
        str(item.get("row_id") or "")
        for item in candidate_manifest.get("excluded") or []
        if item.get("reason") == "already_measured"
    }
    coldstart_by_id = {
        str(row.get("manifest_job_id") or ""): row for row in coldstart_rows
    }
    missing = sorted(initial_ids - set(coldstart_by_id))
    if missing:
        raise ValueError(f"joint initial observed rows missing from Gold176: {missing[:3]}")
    initial_points = []
    initial_failure_count = 0
    for row_id in sorted(initial_ids):
        row = coldstart_by_id[row_id]
        if row.get("terminal_status") == SUCCESS:
            initial_points.append({
                **normalize_gold_row(row, expected_backend=backend),
                "evidence_origin": "joint_actual_v3_initial_observed_gold176",
            })
        elif row.get("terminal_status") in FAILURES:
            initial_failure_count += 1
        else:
            raise ValueError(f"joint initial row is not terminal: {row_id}")

    feedback_paths = sorted(
        task_root.glob("round_*/actual_feedback/stage5_feedback_v3_actual.json")
    )
    online_rows = _unique(
        [row for path in feedback_paths for row in _rows(_read(path))]
    )
    online_points = [
        {
            **normalize_actual_feedback_row(row, expected_backend=backend),
            "evidence_origin": "joint_actual_v3_online_feedback",
        }
        for row in online_rows
        if row.get("terminal_status") == SUCCESS
    ]
    online_failures = sum(row.get("terminal_status") in FAILURES for row in online_rows)
    online_terminal = len(online_points) + online_failures
    task_terminal = _read(task_root / "task_budget_terminal.json")
    complete = (
        online_terminal == EXPECTED_ONLINE["joint_shcosearch"]
        and task_terminal.get("status") == "budget_exhausted"
        and int(task_terminal.get("budget_consumed") or 0) == 16
    )
    return {
        "status": "complete" if complete else "running",
        "independent_validation_complete": False,
        "points": initial_points + online_points,
        "outer_genomes": online_terminal,
        "planned_outer_genomes": 16,
        "initial_observed_count": len(initial_points) + initial_failure_count,
        "initial_failure_count": initial_failure_count,
        "failure_count": online_failures,
        "tuning_trials": 64 * online_terminal,
        "evidence_manifests": [str(path) for path in feedback_paths],
        "evidence_manifest_sha256": {
            str(path): _sha(path) for path in feedback_paths
        },
    }


def _nested_metric(payload: Any, names: set[str]) -> float:
    found = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in names and isinstance(child, (int, float)) and not isinstance(child, bool):
                    found.append(float(child))
                visit(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for child in value:
                visit(child)

    visit(payload)
    finite = [value for value in found if math.isfinite(value)]
    if not finite:
        raise ValueError(f"metric not found: {sorted(names)}")
    return finite[0]


def _verified_tvm_default_schedule_repair(
    formal_root: Path,
) -> tuple[dict[str, Any], Path] | None:
    audit_path = (
        formal_root
        / "tvm/default_schedule_epoch23_repair_v1/repair_audit.json"
    )
    if not audit_path.is_file():
        return None
    audit = _read(audit_path)
    arms = audit.get("arms") or {}
    if (
        audit.get("status") != "passed"
        or audit.get("checkpoint_epoch") != 23
        or set(arms) != {"default", "tuned"}
        or len(audit.get("repeat_evidence") or []) != 3
    ):
        return None
    for policy in ("default", "tuned"):
        arm = arms[policy]
        ap_path = Path(str(arm.get("path") or ""))
        if not ap_path.is_file() or _sha(ap_path) != arm.get("sha256"):
            return None
    for repeat in audit["repeat_evidence"]:
        files = repeat.get("files") or []
        if not files:
            return None
        for item in files:
            path = Path(str(item.get("path") or ""))
            if not path.is_file() or _sha(path) != item.get("sha256"):
                return None
    return audit, audit_path


def _tvm_default_baseline(formal_root: Path) -> dict[str, Any] | None:
    verified = _verified_tvm_default_schedule_repair(formal_root)
    if verified is None:
        return None
    audit, audit_path = verified
    arm = audit["arms"]["default"]
    return {
        "backend": "tvm",
        "schedule_policy": str(arm["schedule_policy"]),
        "evidence_origin": "stage6_tvm_default_zero_trial_epoch23_measurement",
        "config": [64, 128, 256, "fp32"],
        "AP70": float(arm["ap70"]),
        "latency_ms": float(arm["latency_median_ms"]),
        "energy_j": float(arm["energy_median_j"]),
        "tuning_trials": 0,
        "evidence_path": str(audit_path),
        "evidence_sha256": _sha(audit_path),
    }


def _schedule_arm(
    formal_root: Path, backend: str, *, baseline_ap70: float
) -> dict[str, Any]:
    if backend == "tvm":
        verified = _verified_tvm_default_schedule_repair(formal_root)
        if verified is not None:
            audit, audit_path = verified
            tuned = audit["arms"]["tuned"]
            ap_path = Path(str(tuned["path"]))
            point = {
                "manifest_job_id": "stage6|pyramid|schedule_only|tvm|fp32",
                "config": [64, 128, 256, "fp32"],
                "AP70": float(tuned["ap70"]),
                "latency_ms": float(tuned["latency_median_ms"]),
                "energy_j": float(tuned["energy_median_j"]),
                "terminal_status": SUCCESS,
                "independent_validation_passed": True,
                "evidence_sha_verified": True,
                "actual_graph_features_verified": None,
                "evidence_files": {
                    "repair_audit": str(audit_path),
                    "ap": str(ap_path),
                },
                "evidence_file_sha256": {
                    str(audit_path): _sha(audit_path),
                    str(ap_path): _sha(ap_path),
                },
                "evidence_origin": "stage6_tvm_schedule_epoch23_repair_measurement",
            }
            return {
                "status": "complete",
                "evidence_origin": "stage6_tvm_schedule_epoch23_repair_measurement",
                "independent_validation_complete": True,
                "embedded_independent_validation": True,
                "points": [point],
                "outer_genomes": 1,
                "planned_outer_genomes": 1,
                "failure_count": 0,
                "tuning_trials": 64,
            }
    root = formal_root / backend / "schedule_only"
    performance_path = root / "performance_result.json"
    ap_path = root / "ap/full_1789/full_ap_eval_report.json"
    repair_audit_path = (
        formal_root / "trt/schedule_only_epoch23_repair_v1/repair_audit.json"
    )
    repair_audit = None
    if backend == "trt" and repair_audit_path.is_file():
        repair_audit = _read(repair_audit_path)
        repeats = repair_audit.get("performance_repeats") or []
        if (
            repair_audit.get("status") == "passed"
            and repair_audit.get("checkpoint_epoch") == 23
            and repair_audit.get("processed_samples") == 1789
            and len(repeats) == 3
        ):
            candidate_ap = Path(str(repair_audit.get("ap_report_path") or ""))
            candidate_performance = Path(str(repeats[0].get("path") or ""))
            if (
                candidate_ap.is_file()
                and _sha(candidate_ap) == repair_audit.get("ap_report_sha256")
                and candidate_performance.is_file()
                and _sha(candidate_performance) == repeats[0].get("sha256")
            ):
                ap_path = candidate_ap
                performance_path = candidate_performance
    if not (performance_path.is_file() and ap_path.is_file()):
        return {
            "status": "running",
            "evidence_origin": "stage6_new_control_measurement",
            "independent_validation_complete": False,
            "points": [],
            "outer_genomes": 0,
            "planned_outer_genomes": 1,
            "failure_count": 0,
            "tuning_trials": 64,
        }
    performance = _read(performance_path)
    ap = _read(ap_path)
    source = ap.get("ap") if isinstance(ap.get("ap"), Mapping) else ap
    measured_ap70 = _nested_metric(source, {"ap70", "AP70"})
    if measured_ap70 < baseline_ap70 - 0.10:
        return {
            "status": "complete_failure",
            "evidence_origin": "stage6_new_control_measurement",
            "independent_validation_complete": True,
            "failure_evidence_sha_verified": True,
            "failure_reason": "base_fp32_backend_numerical_contract_failure",
            "points": [],
            "outer_genomes": 1,
            "planned_outer_genomes": 1,
            "failure_count": 1,
            "tuning_trials": 64,
            "measured_ap70": measured_ap70,
            "baseline_ap70": baseline_ap70,
            "evidence_file_sha256": {
                str(performance_path): _sha(performance_path),
                str(ap_path): _sha(ap_path),
            },
        }
    point = {
        "manifest_job_id": f"stage6|pyramid|schedule_only|{backend}|fp32",
        "config": [64, 128, 256, "fp32"],
        "AP70": measured_ap70,
        "latency_ms": _nested_metric(
            performance,
            {"lat_p50_ms", "latency_ms", "latency_p50_ms", "latency_ms_p50"},
        ),
        "energy_j": _nested_metric(
            performance, {"energy_j", "joules_per_inference", "joule_per_inference"}
        ),
        "terminal_status": SUCCESS,
        "independent_validation_passed": False,
        "evidence_sha_verified": True,
        "actual_graph_features_verified": None,
        "evidence_files": {
            "performance": str(performance_path),
            "ap": str(ap_path),
            **(
                {"repair_audit": str(repair_audit_path)}
                if repair_audit is not None
                else {}
            ),
        },
        "evidence_file_sha256": {
            str(performance_path): _sha(performance_path),
            str(ap_path): _sha(ap_path),
            **(
                {str(repair_audit_path): _sha(repair_audit_path)}
                if repair_audit is not None
                else {}
            ),
        },
        "evidence_origin": "stage6_new_control_measurement",
    }
    return {
        "status": "complete",
        "evidence_origin": "stage6_new_control_measurement",
        "independent_validation_complete": False,
        "points": [point],
        "outer_genomes": 1,
        "planned_outer_genomes": 1,
        "failure_count": 0,
        "tuning_trials": 64,
    }


def _tune_then_compress_arm(formal_root: Path, backend: str) -> dict[str, Any]:
    paths = sorted(
        (formal_root / backend / "tune_then_compress/attempts").glob("attempt_*.json")
    )
    attempts = [_read(path) for path in paths]
    success = [
        item
        for item in attempts
        if (
            item.get("terminal_status") == "transferred_success"
            and item.get("full_transfer") is True
        )
        or item.get("status") in {"success", "ready", "transfer_success"}
        or item.get("run_success") is True
    ]
    terminal_failures = [
        item
        for item in attempts
        if (
            item.get("terminal_status") == "feasibility_failure"
            and item.get("full_transfer") is False
        )
        or item.get("status") in {
            "failed",
            "feasibility_failure",
            "transfer_failure",
            "incompatible",
        }
        or item.get("build_success") is False
        or item.get("transfer_success") is False
    ]
    all_terminal = len(attempts) == 16 and len(success) + len(terminal_failures) == 16
    if success:
        status = "running"
    elif all_terminal:
        status = "complete_failure"
    else:
        status = "running"
    return {
        "status": status,
        "evidence_origin": "stage6_new_control_measurement",
        "failure_reason": (
            "frozen_backend_policy_transfer_failed_for_all_genomes"
            if status == "complete_failure"
            else None
        ),
        "independent_validation_complete": status == "complete_failure",
        "failure_evidence_sha_verified": all_terminal and not success,
        "points": [],
        "outer_genomes": len(attempts),
        "planned_outer_genomes": 16,
        "failure_count": len(terminal_failures),
        "tuning_trials": 64,
        "attempt_evidence_sha256": {str(path): _sha(path) for path in paths},
        "unmeasured_transfer_success_count": len(success),
    }


def _winner(points: Sequence[Mapping[str, Any]], floor: float) -> str | None:
    eligible = [
        point
        for point in points
        if point.get("evidence_sha_verified") is True
        and point.get("terminal_status") == SUCCESS
        and float(point.get("AP70") or -math.inf) >= floor
    ]
    if not eligible:
        return None
    minimum = min(float(point["latency_ms"]) for point in eligible)
    close = [
        point
        for point in eligible
        if float(point["latency_ms"]) <= minimum * 1.01
    ]
    selected = min(
        close, key=lambda point: (float(point["energy_j"]), float(point["latency_ms"]))
    )
    return str(selected["manifest_job_id"])


def _fastest(points: Sequence[Mapping[str, Any]]) -> str | None:
    eligible = [
        point
        for point in points
        if point.get("evidence_sha_verified") is True
        and point.get("terminal_status") == SUCCESS
    ]
    if not eligible:
        return None
    minimum = min(float(point["latency_ms"]) for point in eligible)
    close = [
        point
        for point in eligible
        if float(point["latency_ms"]) <= minimum * 1.01
    ]
    selected = min(
        close, key=lambda point: (float(point["energy_j"]), float(point["latency_ms"]))
    )
    return str(selected["manifest_job_id"])


def _apply_validation(
    arms: Mapping[str, Mapping[str, Any]],
    validation_index: Mapping[tuple[str, str], Mapping[str, Any]],
    baseline_ap70: float,
) -> dict[str, dict[str, Any]]:
    result = {}
    floors = (baseline_ap70 - 0.05, baseline_ap70 - 0.10)
    for arm_id, source in arms.items():
        summary = dict(source)
        points = list(summary.get("points") or [])
        targets = list(
            dict.fromkeys(
                target
                for floor in floors
                if (target := (_winner(points, floor) or _fastest(points)))
            )
        )
        summary["independent_validation_target_ids"] = targets
        if summary.get("embedded_independent_validation") is True:
            result[arm_id] = summary
            continue
        summary["points"] = apply_independent_validation(
            points, validation_index, arm_id=arm_id
        )
        if summary.get("status") == "complete":
            summary["independent_validation_complete"] = bool(targets) and all(
                (arm_id, target) in validation_index for target in targets
            )
        result[arm_id] = summary
    return result


def _baseline(formal_root: Path) -> dict[str, Any]:
    path = formal_root / "stage6_native_fp32_baseline_with_ap_v3.json"
    payload = _read(path)
    ap_path = Path(str(payload["ap_report_path"]))
    if _sha(ap_path) != payload.get("ap_report_sha256"):
        raise ValueError("native baseline AP SHA mismatch")
    if int(payload.get("ap_num_samples") or 0) != 1789:
        raise ValueError("native baseline full AP contract mismatch")
    return {
        "config": [64, 128, 256, "fp32"],
        "AP70": float(payload["ap70"]),
        "latency_ms": float(payload["latency_p50_ms"]),
        "energy_j": float(payload["energy_j"]),
        "independent_repeat_count": int(payload["independent_repeat_count"]),
        "evidence_path": str(path),
        "evidence_sha256": _sha(path),
        "ap_report_path": str(ap_path),
        "ap_report_sha256": _sha(ap_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--joint-root", type=Path, required=True)
    parser.add_argument("--coldstart-rows-json", type=Path, required=True)
    parser.add_argument("--independent-audit", type=Path, action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = _baseline(args.formal_root)
    coldstart = _rows(_read(args.coldstart_rows_json))
    audits = [_read(path) for path in args.independent_audit]
    validation_index = build_independent_validation_index(audits) if audits else {}
    backends = {}
    normalizations = {}
    for backend in ("tvm", "trt"):
        raw_arms = {
            "compression_only": _actual_arm(
                args.formal_root, backend, "compression_only"
            ),
            "schedule_only": _schedule_arm(
                args.formal_root, backend, baseline_ap70=float(baseline["AP70"])
            ),
            "compress_then_tune": _actual_arm(
                args.formal_root, backend, "compress_then_tune"
            ),
            "tune_then_compress": _tune_then_compress_arm(
                args.formal_root, backend
            ),
            "joint_shcosearch": _joint_arm(
                backend=backend,
                joint_root=args.joint_root,
                coldstart_rows=coldstart,
            ),
        }
        hv = attach_common_hypervolume(raw_arms)
        backends[backend] = _apply_validation(
            hv["arms"], validation_index, float(baseline["AP70"])
        )
        normalizations[backend] = hv["normalization"]
    bundle = {
        "schema_version": "stage6_paper_evidence_bundle_v1",
        "baseline": baseline,
        "backends": backends,
        "common_hv_normalization": normalizations,
        "independent_validation_audits": [
            {"path": str(path), "sha256": _sha(path)}
            for path in args.independent_audit
        ],
    }
    tvm_baseline = _tvm_default_baseline(args.formal_root)
    if tvm_baseline is not None:
        bundle["backend_baselines"] = {
            "trt": baseline,
            "tvm": tvm_baseline,
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "arm_statuses": {
                    backend: {
                        arm: summary["status"]
                        for arm, summary in arms.items()
                    }
                    for backend, arms in backends.items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
