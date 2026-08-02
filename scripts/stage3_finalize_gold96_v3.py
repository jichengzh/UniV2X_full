#!/usr/bin/env python3
"""Finalize Stage3 Gold96 evidence into row, group-audit, and summary outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA = "stage3_gold_coldstart96_manifest_v3"
OUTPUT_SCHEMA = "stage3_gold96_final_v3"
AP_TERMINAL_RECORD_TYPES = {"terminal_event", "job_terminal"}
FAILURE_STATUSES = {"feasibility_failure", "numerical_feasibility_failure"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    item = Path(path)
    if not item.is_file():
        return []
    return [json.loads(line) for line in item.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _first_finite(payload: Mapping[str, Any], paths: Sequence[tuple[str, ...]]) -> float | None:
    for path in paths:
        value: Any = payload
        for key in path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(key)
        if _finite(value):
            return float(value)
    return None


def _performance_metrics(result_json: Any) -> tuple[float, float, str] | None:
    if not isinstance(result_json, str) or not result_json:
        return None
    path = Path(result_json)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    latency = _first_finite(payload, (
        ("lat_p50_ms",), ("latency_ms",), ("latency", "latency_ms_p50"), ("latency", "lat_p50_ms"),
    ))
    energy = _first_finite(payload, (
        ("energy_j",), ("energy", "energy_j"), ("energy", "joules"),
        ("energy", "joules_per_inference"), ("energy", "joule_per_inference"), ("energy", "energy_J"),
    ))
    if latency is None or energy is None:
        return None
    return latency, energy, sha256_file(path)


def _ap_values(report: Mapping[str, Any]) -> tuple[float, float, float] | None:
    nested = report.get("ap")
    sources = (nested, report) if isinstance(nested, Mapping) else (report,)
    values: list[float] = []
    for key in ("ap30", "ap50", "ap70"):
        value = next((source.get(key) for source in sources if _finite(source.get(key))), None)
        if not _finite(value):
            return None
        values.append(float(value))
    return values[0], values[1], values[2]


def _read_report(row: Mapping[str, Any]) -> tuple[Path, Mapping[str, Any], str] | None:
    report_path = row.get("report_path") or row.get("report_json")
    if not isinstance(report_path, str):
        return None
    path = Path(report_path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    return path, payload, sha256_file(path)


def _valid_ap_report(source: Mapping[str, Any], report: Mapping[str, Any], stage: str) -> bool:
    sample_count = report.get("processed_samples")
    fallback = report.get("fallback_samples", 0)
    failed = report.get("failed_samples", 0)
    if report.get("status") != "success" or fallback != 0 or failed != 0:
        return False
    if stage == "full" and sample_count != 1789:
        return False
    if stage == "sanity" and (not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 16):
        return False

    if source.get("model") == "codriving":
        gates = report.get("gates")
        gate = "full_1789" if stage == "full" else "sanity_16"
        return isinstance(gates, Mapping) and gates.get(gate) is True
    if source.get("model") == "pyramid" and stage == "full":
        if source.get("dispatch_key") == "tvm_auto":
            legacy_passed = report.get("smoke_gate_passed") is True
            gates = report.get("gates")
            numeric_gate_passed = (
                isinstance(gates, Mapping) and gates.get("full_1789") is True
                and report.get("ap_row_allowed") is True
                and report.get("feasibility_blockers") in (None, [])
            )
            return report.get("ap_measured") is True and (legacy_passed or numeric_gate_passed)
        return report.get("engine_ap_claim") is True
    return source.get("model") == "pyramid"


def _valid_stage5_numerical_failure_report(
    source: Mapping[str, Any], report: Mapping[str, Any]
) -> bool:
    reasons = report.get("failure_reasons")
    outputs = report.get("numeric_outputs")
    return (
        source.get("model") == "codriving"
        and source.get("dispatch_key") == "tvm_auto"
        and (source.get("q_mode") or source.get("q")) == "int8"
        and report.get("status") == "numerical_feasibility_failure"
        and report.get("processed_samples") == 16
        and report.get("engine_samples") == 16
        and report.get("engine_accounting_valid") is True
        and report.get("fallback_samples") == 0
        and report.get("failed_samples") == 0
        and report.get("ap_measured") is False
        and report.get("gates", {}).get("sanity_16") is False
        and isinstance(reasons, list)
        and bool(reasons)
        and all(isinstance(reason, str) and reason for reason in reasons)
        and isinstance(outputs, Mapping)
        and len(outputs) == 3
        and any(isinstance(value, Mapping) and value.get("passed") is False for value in outputs.values())
        and not any(key in report for key in ("ap", "ap30", "ap50", "ap70"))
    )


def _terminal_performance(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    terminal = [row for row in rows if row.get("status") in {"success", "confirmed_failure"}]
    return terminal[-1] if terminal else None


def _base_row(source: Mapping[str, Any], *, output_schema: str = OUTPUT_SCHEMA) -> dict[str, Any]:
    return {
        "schema_version": output_schema,
        "manifest_job_id": str(source["job_id"]),
        "group_id": str(source["group_id"]),
        "model": source.get("model"),
        "width": list(source.get("width") or []),
        "q_mode": source.get("q_mode") or source.get("q"),
        "capability_profile_id": source.get("capability_profile_id") or source.get("profile"),
        "dispatch_key": source.get("dispatch_key"),
        "terminal_status": None,
        "latency_ms": None,
        "energy_j": None,
        "performance_result_json": None,
        "performance_result_sha256": None,
        "ap30": None,
        "ap50": None,
        "ap70": None,
        "ap_report_path": None,
        "ap_report_sha256": None,
        "failure_reason": None,
    }


def _finalize_row(
    source: Mapping[str, Any],
    performance_rows: Sequence[Mapping[str, Any]],
    ap_rows: Sequence[Mapping[str, Any]],
    *,
    output_schema: str = OUTPUT_SCHEMA,
) -> dict[str, Any]:
    output = _base_row(source, output_schema=output_schema)
    performance = _terminal_performance(performance_rows)
    if performance is None:
        output["terminal_status"] = "pending_performance"
        return output
    if performance.get("status") == "confirmed_failure":
        failure_reason = performance.get("failure_reason")
        if not isinstance(failure_reason, str) or not failure_reason.strip():
            executor_reasons = performance.get("failure_reasons")
            normalized_reasons = (
                [str(reason).strip() for reason in executor_reasons if str(reason).strip()]
                if isinstance(executor_reasons, list)
                else []
            )
            suffix = ",".join(normalized_reasons)
            failure_reason = "confirmed_backend_performance_failure"
            if suffix:
                failure_reason = f"{failure_reason}:{suffix}"
        output["terminal_status"] = "feasibility_failure"
        output["failure_reason"] = failure_reason
        return output

    metrics = _performance_metrics(performance.get("result_json"))
    if metrics is None:
        output["terminal_status"] = "blocked_performance_evidence"
        output["performance_result_json"] = performance.get("result_json")
        return output
    output["latency_ms"], output["energy_j"], output["performance_result_sha256"] = metrics
    output["performance_result_json"] = performance.get("result_json")

    terminal_ap = [row for row in ap_rows if row.get("record_type") in AP_TERMINAL_RECORD_TYPES]
    full_successes = [row for row in terminal_ap if row.get("stage") == "full" and row.get("status") == "success"]
    full_numerical_skips = [
        row for row in terminal_ap
        if row.get("stage") == "full"
        and row.get("status") == "skipped_numerical_feasibility"
        and row.get("failure_reason") == "numerical_feasibility_failure"
    ]
    if full_successes:
        full = full_successes[-1]
        evidence = _read_report(full)
        if evidence is None:
            output["terminal_status"] = "blocked_ap_evidence"
            return output
        report_path, report, report_sha = evidence
        ap = _ap_values(report)
        if ap is None or not _valid_ap_report(source, report, "full"):
            output["terminal_status"] = "blocked_ap_evidence"
            return output
        output["terminal_status"] = "measured_success_gold"
        output["ap30"], output["ap50"], output["ap70"] = ap
        output["ap_report_path"] = str(report_path)
        output["ap_report_sha256"] = report_sha
        return output

    if any(row.get("stage") == "full" and row.get("status") == "failed" for row in terminal_ap):
        output["terminal_status"] = "blocked_ap_evidence"
        return output

    sanity = [row for row in terminal_ap if row.get("stage") == "sanity" and row.get("status") in {"success", "failed"}]
    if not sanity:
        output["terminal_status"] = "pending_ap"
        return output
    latest_sanity = sanity[-1]
    evidence = _read_report(latest_sanity)
    if evidence is None:
        output["terminal_status"] = "blocked_ap_evidence"
    elif latest_sanity.get("status") == "failed":
        report_path, report, report_sha = evidence
        strict_stage5 = output_schema == "stage5_feedback_row_v2"
        matching_full_skip = any(
            row.get("report_path") == str(report_path)
            and row.get("report_sha256") == report_sha
            for row in full_numerical_skips
        )
        valid_failure = (
            latest_sanity.get("failure_reason") == "numerical_feasibility_failure"
            and _valid_stage5_numerical_failure_report(source, report)
            and (not strict_stage5 or matching_full_skip)
        )
        if strict_stage5 and not valid_failure:
            output["terminal_status"] = "blocked_ap_evidence"
        else:
            output["terminal_status"] = "numerical_feasibility_failure"
            output["failure_reason"] = latest_sanity.get("failure_reason")
            output["ap_report_path"] = str(report_path)
            output["ap_report_sha256"] = report_sha
    elif not _valid_ap_report(source, evidence[1], "sanity"):
        output["terminal_status"] = "blocked_ap_evidence"
    else:
        output["terminal_status"] = "pending_full_ap"
    return output


def _validate_inputs(
    manifest: Mapping[str, Any],
    ap_plan_rows: Sequence[Mapping[str, Any]],
    *,
    manifest_schema: str = MANIFEST_SCHEMA,
    expected_rows: int = 96,
    expected_groups: int = 24,
) -> list[Mapping[str, Any]]:
    jobs = manifest.get("jobs")
    if manifest.get("schema_version") != manifest_schema or not isinstance(jobs, list) or len(jobs) != expected_rows:
        raise ValueError(f"expected {manifest_schema} with exactly {expected_rows} jobs")
    ids = [str(row.get("job_id") or "") for row in jobs if isinstance(row, Mapping)]
    if len(ids) != expected_rows or len(set(ids)) != expected_rows:
        raise ValueError(f"manifest jobs must have {expected_rows} unique job_id values")
    groups = Counter(str(row.get("group_id") or "") for row in jobs)
    if len(groups) != expected_groups or set(groups.values()) != {4}:
        raise ValueError(f"manifest must contain {expected_groups} groups with exactly four rows each")
    plan_ids = [str(row.get("manifest_job_id") or "") for row in ap_plan_rows]
    if len(plan_ids) != expected_rows or len(set(plan_ids)) != expected_rows or set(plan_ids) != set(ids):
        raise ValueError("AP plan must map exactly once to every manifest job")
    return jobs


def finalize_gold_dataset(
    manifest: Mapping[str, Any],
    *,
    ap_plan_rows: Sequence[Mapping[str, Any]],
    performance_state_rows: Sequence[Mapping[str, Any]],
    ap_state_rows: Sequence[Mapping[str, Any]],
    manifest_schema: str,
    output_schema: str,
    expected_rows: int,
    expected_groups: int,
) -> dict[str, Any]:
    jobs = _validate_inputs(
        manifest,
        ap_plan_rows,
        manifest_schema=manifest_schema,
        expected_rows=expected_rows,
        expected_groups=expected_groups,
    )
    plan_by_manifest_id = {str(row["manifest_job_id"]): row for row in ap_plan_rows}
    performance_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for row in performance_state_rows:
        performance_by_id.setdefault(str(row.get("manifest_job_id") or row.get("job_id") or ""), []).append(row)
    ap_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for row in ap_state_rows:
        ap_by_id.setdefault(str(row.get("manifest_job_id") or row.get("job_id") or ""), []).append(row)

    rows = [
        _finalize_row(
            source,
            performance_by_id.get(
                str(plan_by_manifest_id[str(source["job_id"])].get("performance_job_id") or source["job_id"]),
                (),
            ),
            ap_by_id.get(str(source["job_id"]), ()),
            output_schema=output_schema,
        )
        for source in jobs
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["group_id"], []).append(row)
    group_audit = [
        {
            "group_id": group_id,
            "row_count": len(group_rows),
            "manifest_job_ids": [row["manifest_job_id"] for row in group_rows],
            "terminal_statuses": [row["terminal_status"] for row in group_rows],
            "all_terminal": all(bool(row["terminal_status"]) for row in group_rows),
        }
        for group_id, group_rows in grouped.items()
    ]
    summary = {
        "measured": sum(row["terminal_status"] == "measured_success_gold" for row in rows),
        "failure": sum(row["terminal_status"] in FAILURE_STATUSES for row in rows),
        "pending": sum(row["terminal_status"] not in FAILURE_STATUSES | {"measured_success_gold"} for row in rows),
        "total": len(rows),
    }
    return {"rows": rows, "group_audit": group_audit, "summary": summary}


def finalize_gold96(
    manifest: Mapping[str, Any],
    *,
    ap_plan_rows: Sequence[Mapping[str, Any]],
    performance_state_rows: Sequence[Mapping[str, Any]],
    ap_state_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return finalize_gold_dataset(
        manifest,
        ap_plan_rows=ap_plan_rows,
        performance_state_rows=performance_state_rows,
        ap_state_rows=ap_state_rows,
        manifest_schema=MANIFEST_SCHEMA,
        output_schema=OUTPUT_SCHEMA,
        expected_rows=96,
        expected_groups=24,
    )


def _csv_value(value: Any) -> Any:
    return json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value


def write_dataset_outputs(
    result: Mapping[str, Any],
    output_dir: str | Path,
    *,
    file_prefix: str,
    output_schema: str,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = list(result["rows"])
    (output / f"{file_prefix}_final.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / f"{file_prefix}_final.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    with (output / f"{file_prefix}_final.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: _csv_value(row.get(key)) for key in fieldnames}
            for row in rows
        )
    audit = {"schema_version": output_schema, "groups": result["group_audit"], "summary": result["summary"]}
    (output / f"{file_prefix}_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_outputs(result: Mapping[str, Any], output_dir: str | Path) -> None:
    write_dataset_outputs(result, output_dir, file_prefix="gold96", output_schema=OUTPUT_SCHEMA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--ap-plan-jsonl", type=Path, required=True)
    parser.add_argument("--performance-state-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--ap-state-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    result = finalize_gold96(
        manifest,
        ap_plan_rows=read_jsonl(args.ap_plan_jsonl),
        performance_state_rows=[row for path in args.performance_state_jsonl for row in read_jsonl(path)],
        ap_state_rows=[row for path in args.ap_state_jsonl for row in read_jsonl(path)],
    )
    write_outputs(result, args.output_dir)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
