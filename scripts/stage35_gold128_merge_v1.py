#!/usr/bin/env python3
"""Merge Gold96 and Gold32 into a grouped, leakage-safe Gold128 dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


OUTPUT_SCHEMA = "stage35_gold128_final_v1"
MANIFEST_SCHEMA = "stage35_gold128_manifest_v1"
ARMS = {
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_pool(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    name: str,
    expected_rows: int,
    expected_groups: int,
    expected_locked_groups: int,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, list[Mapping[str, Any]]]]:
    jobs = manifest.get("jobs")
    if len(rows) != expected_rows or not isinstance(jobs, list) or len(jobs) != expected_rows:
        raise ValueError(f"{name} must contain exactly {expected_rows} rows and jobs")
    jobs_by_id = {str(job.get("job_id")): job for job in jobs if isinstance(job, Mapping)}
    rows_by_id = {str(row.get("manifest_job_id")): row for row in rows}
    if len(jobs_by_id) != expected_rows or set(jobs_by_id) != set(rows_by_id):
        raise ValueError(f"{name} result/manifest job IDs must be unique and identical")
    binding_fields = (
        "group_id", "model", "width", "q_mode", "dispatch_key", "capability_profile_id"
    )
    for job_id, row in rows_by_id.items():
        job = jobs_by_id[job_id]
        mismatches = [field for field in binding_fields if row.get(field) != job.get(field)]
        if mismatches:
            raise ValueError(
                f"{name} row/manifest binding mismatch for {job_id}: {mismatches}"
            )

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("group_id")), []).append(row)
    if len(grouped) != expected_groups:
        raise ValueError(f"{name} must contain exactly {expected_groups} groups")
    for group_id, group_rows in grouped.items():
        arms = {(str(row.get("dispatch_key")), str(row.get("q_mode"))) for row in group_rows}
        if len(group_rows) != 4 or arms != ARMS:
            raise ValueError(f"{name} group {group_id} is not a complete four-arm group")

    group_splits: dict[str, str] = {}
    for job in jobs:
        group_id = str(job["group_id"])
        split = str(job.get("split"))
        if split not in {"train", "locked_holdout"}:
            raise ValueError(f"{name} group {group_id} has invalid split {split}")
        if group_id in group_splits and group_splits[group_id] != split:
            raise ValueError(f"{name} group {group_id} crosses split boundaries")
        group_splits[group_id] = split
    if sum(split == "locked_holdout" for split in group_splits.values()) != expected_locked_groups:
        raise ValueError(f"{name} must contain exactly {expected_locked_groups} locked groups")
    group_models = {
        str(job["group_id"]): str(job["model"])
        for job in jobs
    }
    models = sorted(set(group_models.values()))
    expected_locked_per_model = expected_locked_groups // len(models)
    expected_train_per_model = (expected_groups - expected_locked_groups) // len(models)
    split_by_model = Counter(
        (group_models[group_id], split) for group_id, split in group_splits.items()
    )
    if any(
        split_by_model[(model, "locked_holdout")] != expected_locked_per_model
        or split_by_model[(model, "train")] != expected_train_per_model
        for model in models
    ):
        raise ValueError(f"{name} per-model split balance is invalid: {dict(split_by_model)}")

    for row in rows:
        status = row.get("terminal_status")
        if status == "measured_success_gold":
            if not all(_finite(row.get(key)) for key in ("latency_ms", "energy_j", "ap30", "ap50", "ap70")):
                raise ValueError(f"{name} measured row lacks finite metrics: {row.get('manifest_job_id')}")
            if not all(isinstance(row.get(key), str) and len(str(row[key])) == 64 for key in ("performance_result_sha256", "ap_report_sha256")):
                raise ValueError(f"{name} measured row lacks evidence hashes: {row.get('manifest_job_id')}")
        elif status != "feasibility_failure":
            raise ValueError(f"{name} has non-terminal row: {row.get('manifest_job_id')}")
        if group_splits[str(row["group_id"])] == "locked_holdout" and status != "measured_success_gold":
            raise ValueError(f"{name} locked holdout row must be measured: {row.get('manifest_job_id')}")
    return jobs_by_id, grouped


def merge_gold128(
    gold96_rows: Sequence[Mapping[str, Any]],
    manifest96: Mapping[str, Any],
    gold32_rows: Sequence[Mapping[str, Any]],
    manifest32: Mapping[str, Any],
) -> dict[str, Any]:
    jobs96, groups96 = _validate_pool(
        gold96_rows, manifest96, name="gold96", expected_rows=96, expected_groups=24, expected_locked_groups=4
    )
    jobs32, groups32 = _validate_pool(
        gold32_rows, manifest32, name="gold32", expected_rows=32, expected_groups=8, expected_locked_groups=2
    )
    overlap = set(groups96) & set(groups32)
    if overlap:
        raise ValueError(f"Gold96/Gold32 group overlap is forbidden: {sorted(overlap)}")

    output_rows: list[dict[str, Any]] = []
    output_jobs: list[dict[str, Any]] = []
    for source_pool, rows, jobs_by_id in (
        ("gold96", gold96_rows, jobs96),
        ("gold32", gold32_rows, jobs32),
    ):
        for source in rows:
            job_id = str(source["manifest_job_id"])
            job = jobs_by_id[job_id]
            output_rows.append({
                **dict(source),
                "schema_version": OUTPUT_SCHEMA,
                "split": str(job["split"]),
                "width_stratum": str(job.get("width_stratum") or "unspecified"),
                "source_pool": source_pool,
            })
            output_jobs.append({
                **dict(job),
                "schema_version": MANIFEST_SCHEMA,
                "source_pool": source_pool,
            })

    group_split = {
        str(job["group_id"]): str(job["split"])
        for job in output_jobs
    }
    pilot_groups = [str(group) for group in manifest96.get("pilot_group_ids", [])]
    if len(pilot_groups) != 2 or not set(pilot_groups) <= set(group_split):
        raise ValueError("Gold96 must provide two valid pilot/reference groups")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "jobs": output_jobs,
        "pilot_group_ids": pilot_groups,
        "source_manifests": {
            "gold96_schema": manifest96.get("schema_version"),
            "gold32_schema": manifest32.get("schema_version"),
        },
    }
    split_counts = Counter(group_split.values())
    status_counts = Counter(str(row["terminal_status"]) for row in output_rows)
    audit = {
        "schema_version": OUTPUT_SCHEMA,
        "rows": len(output_rows),
        "groups": len(group_split),
        "source_rows": dict(Counter(str(row["source_pool"]) for row in output_rows)),
        "split_groups": dict(split_counts),
        "status_rows": dict(status_counts),
        "model_rows": dict(Counter(str(row["model"]) for row in output_rows)),
        "backend_rows": dict(Counter(str(row["dispatch_key"]) for row in output_rows)),
        "q_mode_rows": dict(Counter(str(row["q_mode"]) for row in output_rows)),
    }
    if audit["rows"] != 128 or audit["groups"] != 32 or audit["split_groups"] != {"train": 26, "locked_holdout": 6}:
        raise ValueError(f"Gold128 closure invariant failed: {audit}")
    return {"rows": output_rows, "manifest": manifest, "audit": audit}


def _csv_value(value: Any) -> Any:
    return json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value


def write_outputs(result: Mapping[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = list(result["rows"])
    (output / "gold128_final.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "gold128_final.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "gold128_final.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in rows)
    (output / "gold128_manifest.json").write_text(
        json.dumps(result["manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "gold128_audit.json").write_text(
        json.dumps(result["audit"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold96-json", type=Path, required=True)
    parser.add_argument("--manifest96-json", type=Path, required=True)
    parser.add_argument("--gold32-json", type=Path, required=True)
    parser.add_argument("--manifest32-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = merge_gold128(
        json.loads(args.gold96_json.read_text(encoding="utf-8")),
        json.loads(args.manifest96_json.read_text(encoding="utf-8")),
        json.loads(args.gold32_json.read_text(encoding="utf-8")),
        json.loads(args.manifest32_json.read_text(encoding="utf-8")),
    )
    result["audit"]["source_sha256"] = {
        "gold96": sha256_file(args.gold96_json),
        "manifest96": sha256_file(args.manifest96_json),
        "gold32": sha256_file(args.gold32_json),
        "manifest32": sha256_file(args.manifest32_json),
    }
    write_outputs(result, args.output_dir)
    print(json.dumps(result["audit"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
