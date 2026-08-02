#!/usr/bin/env python3
"""Audit four cross-time full-AP repeats against frozen Gold128 AP values."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from stage3_execute_ap_plan_v3 import plan_fingerprint


SCHEMA = "stage35_gold128_ap_repeat_audit_v1"
AP_KEYS = ("ap30", "ap50", "ap70")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def _two_point_cv(first: float, second: float) -> float:
    values = np.asarray([first, second], dtype=float)
    mean = float(np.mean(values))
    if mean == 0.0:
        return 0.0 if first == second else math.inf
    return float(np.std(values, ddof=1) / abs(mean) * 100.0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_report(state: Mapping[str, Any], job_id: str) -> Mapping[str, Any]:
    path = Path(str(state.get("report_path") or ""))
    digest = str(state.get("report_sha256") or "")
    if not path.is_file() or len(digest) != 64 or _sha256_file(path) != digest:
        raise ValueError(f"AP repeat report evidence mismatch: {job_id}")
    report = json.loads(path.read_text(encoding="utf-8"))
    return report


def _report_is_bound_to_plan(
    plan: Mapping[str, Any], state: Mapping[str, Any], stage: str
) -> bool:
    command = list(map(str, plan.get(f"{stage}_command") or []))
    for flag in ("--export-report-json", "--report-json", "--out-json"):
        if flag in command:
            expected = Path(command[command.index(flag) + 1]).resolve()
            return Path(str(state.get("report_path") or "")).resolve() == expected
    return False


def _report_ap(report: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = report.get("ap")
    return nested if isinstance(nested, Mapping) else report


def build_ap_repeat_audit(
    plans: Sequence[Mapping[str, Any]], state_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    plans_by_id = {str(row.get("job_id")): row for row in plans}
    successes: dict[str, Mapping[str, Any]] = {}
    sanity_successes: dict[str, Mapping[str, Any]] = {}
    for row in state_rows:
        job_id = str(row.get("job_id"))
        plan = plans_by_id.get(job_id)
        stage = str(row.get("stage") or "")
        if plan is None or stage not in {"sanity", "full"} or row.get("status") != "success":
            continue
        if row.get("plan_fingerprint") != plan_fingerprint(plan, stage):
            continue
        if stage == "full":
            successes[job_id] = row
        else:
            sanity_successes[job_id] = row
    if len(plans_by_id) != 4 or set(successes) != set(plans_by_id) or set(sanity_successes) != set(plans_by_id):
        raise ValueError(
            "AP repeat audit requires 4 successful sanity and full jobs, "
            f"got plans={len(plans_by_id)} sanity={len(sanity_successes)} full={len(successes)}"
        )

    rows: list[dict[str, Any]] = []
    for job_id, plan in plans_by_id.items():
        state = successes[job_id]
        sanity_state = sanity_successes[job_id]
        source_manifest_job_id = str(plan.get("source_manifest_job_id") or "")
        baseline_ap_report_sha256 = str(plan.get("baseline_ap_report_sha256") or "")
        if not source_manifest_job_id or len(baseline_ap_report_sha256) != 64:
            raise ValueError(f"AP repeat plan lacks Gold provenance: {job_id}")
        if state.get("plan_fingerprint") != plan_fingerprint(plan, "full"):
            raise ValueError(f"stale AP full state: {job_id}")
        if sanity_state.get("plan_fingerprint") != plan_fingerprint(plan, "sanity"):
            raise ValueError(f"stale AP sanity state: {job_id}")
        if not _report_is_bound_to_plan(plan, state, "full"):
            raise ValueError(f"AP full report is not bound to plan: {job_id}")
        if not _report_is_bound_to_plan(plan, sanity_state, "sanity"):
            raise ValueError(f"AP sanity report is not bound to plan: {job_id}")
        artifact = Path(str(plan.get("compiled_artifact_path") or ""))
        if not artifact.is_file() or _sha256_file(artifact) != plan.get("compiled_artifact_digest"):
            raise ValueError(f"AP compiled artifact evidence mismatch: {job_id}")
        report = _verified_report(state, job_id)
        _verified_report(sanity_state, job_id)
        measured = state.get("ap")
        if not isinstance(measured, Mapping):
            raise ValueError(f"AP repeat state lacks AP mapping: {job_id}")
        row: dict[str, Any] = {
            "schema_version": SCHEMA,
            "job_id": job_id,
            "source_manifest_job_id": source_manifest_job_id,
            "repeat_category": plan["repeat_category"],
            "baseline_ap_report_sha256": baseline_ap_report_sha256,
            "compiled_artifact_digest": plan.get("compiled_artifact_digest"),
            "repeat_report_path": state.get("report_path"),
            "repeat_report_sha256": state.get("report_sha256"),
        }
        for key in AP_KEYS:
            baseline = plan.get(f"baseline_{key}")
            repeat = measured.get(key)
            report_value = _report_ap(report).get(key)
            if not _finite(baseline) or not _finite(repeat):
                raise ValueError(f"non-finite {key} repeat evidence: {job_id}")
            if not _finite(report_value) or float(report_value) != float(repeat):
                raise ValueError(f"state/report {key} mismatch: {job_id}")
            baseline_value = float(baseline)
            repeat_value = float(repeat)
            row[f"baseline_{key}"] = baseline_value
            row[f"repeat_{key}"] = repeat_value
            row[f"abs_delta_{key}"] = abs(repeat_value - baseline_value)
            row[f"cv_pct_{key}"] = _two_point_cv(baseline_value, repeat_value)
        rows.append(row)

    metric_summary = {}
    checks = {}
    for key in AP_KEYS:
        absolute = _distribution([float(row[f"abs_delta_{key}"]) for row in rows])
        cv = _distribution([float(row[f"cv_pct_{key}"]) for row in rows])
        metric_summary[key] = {"absolute_delta": absolute, "cv_pct": cv}
        checks[f"{key}_abs_delta_median_le_0p01"] = absolute["median"] <= 0.01
        checks[f"{key}_abs_delta_p90_le_0p03"] = absolute["p90"] <= 0.03
        checks[f"{key}_cv_median_le_3pct"] = cv["median"] <= 3.0
        checks[f"{key}_cv_p90_le_8pct"] = cv["p90"] <= 8.0
    return {
        "schema_version": SCHEMA,
        "terminal_rows": len(rows),
        "metric_summary": metric_summary,
        "checks": checks,
        "category_summary": {str(row["repeat_category"]): {key: row[f"cv_pct_{key}"] for key in AP_KEYS} for row in rows},
        "qualified": all(checks.values()),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ap-plan-jsonl", type=Path, required=True)
    parser.add_argument("--state-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_ap_repeat_audit(
        read_jsonl(args.ap_plan_jsonl),
        [row for path in args.state_jsonl for row in read_jsonl(path)],
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    print(json.dumps({"terminal_rows": result["terminal_rows"], "qualified": result["qualified"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
