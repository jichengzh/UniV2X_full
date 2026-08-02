#!/usr/bin/env python3
"""Merge Gold128 v2 and targeted16 while freezing the six locked groups."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


GOLD128_OUTPUT_SCHEMA = "stage35_gold128_final_v1"
GOLD128_MANIFEST_SCHEMA = "stage35_gold128_manifest_v1"
TARGETED16_OUTPUT_SCHEMA = "stage35_targeted16_final_v1"
TARGETED16_MANIFEST_SCHEMA = "stage35_gold128_targeted_supplement_manifest_v1"
OUTPUT_SCHEMA = "stage35_gold144_final_v1"
MANIFEST_SCHEMA = "stage35_gold144_manifest_v1"
ARMS = {
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
}
METRIC_FIELDS = ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
HASH_FIELDS = ("performance_result_sha256", "ap_report_sha256")
LOCKED_FIELDS = ("manifest_job_id",) + METRIC_FIELDS + HASH_FIELDS
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_evidence(row: Mapping[str, Any], *, name: str) -> None:
    manifest_id = str(row.get("manifest_job_id"))
    if not all(_finite(row.get(field)) for field in METRIC_FIELDS):
        raise ValueError(f"{name} measured row lacks finite metrics: {manifest_id}")
    if not all(
        isinstance(row.get(field), str) and SHA256_PATTERN.fullmatch(str(row[field]))
        for field in HASH_FIELDS
    ):
        raise ValueError(f"{name} measured row lacks valid evidence SHA256 values: {manifest_id}")


def _validate_pool(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    name: str,
    row_schema: str,
    manifest_schema: str,
    expected_rows: int,
    expected_groups: int,
    expected_locked_groups: int,
    require_all_measured: bool,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, list[Mapping[str, Any]]], dict[str, str]]:
    jobs = manifest.get("jobs")
    if manifest.get("schema_version") != manifest_schema:
        raise ValueError(f"{name} expected manifest schema {manifest_schema}")
    if len(rows) != expected_rows or not isinstance(jobs, list) or len(jobs) != expected_rows:
        raise ValueError(f"{name} must contain exactly {expected_rows} rows and jobs")
    if any(not isinstance(row, Mapping) for row in rows) or any(
        not isinstance(job, Mapping) for job in jobs
    ):
        raise ValueError(f"{name} rows and jobs must be objects")
    if any(row.get("schema_version") != row_schema for row in rows):
        raise ValueError(f"{name} expected row schema {row_schema}")

    row_ids = [str(row.get("manifest_job_id") or "") for row in rows]
    job_ids = [str(job.get("job_id") or "") for job in jobs]
    if any(not value for value in row_ids + job_ids):
        raise ValueError(f"{name} manifest IDs must be non-empty")
    if len(set(row_ids)) != expected_rows or len(set(job_ids)) != expected_rows:
        raise ValueError(f"{name} contains duplicate manifest ID values")
    if set(row_ids) != set(job_ids):
        raise ValueError(f"{name} result/manifest IDs must be identical")

    jobs_by_id = {str(job["job_id"]): job for job in jobs}
    binding_fields = (
        "group_id",
        "model",
        "width",
        "q_mode",
        "dispatch_key",
        "capability_profile_id",
        "split",
    )
    for row in rows:
        manifest_id = str(row["manifest_job_id"])
        job = jobs_by_id[manifest_id]
        mismatches = [field for field in binding_fields if row.get(field) != job.get(field)]
        if mismatches:
            raise ValueError(f"{name} row/manifest binding mismatch for {manifest_id}: {mismatches}")

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        group_id = str(row.get("group_id") or "")
        if not group_id:
            raise ValueError(f"{name} row is missing group_id")
        grouped.setdefault(group_id, []).append(row)
    if len(grouped) != expected_groups:
        raise ValueError(f"{name} must contain exactly {expected_groups} groups")
    for group_id, group_rows in grouped.items():
        arms = {
            (str(row.get("dispatch_key") or ""), str(row.get("q_mode") or ""))
            for row in group_rows
        }
        if len(group_rows) != 4 or arms != ARMS:
            raise ValueError(f"{name} group {group_id} is not a complete four-arm group")

    group_splits: dict[str, str] = {}
    for job in jobs:
        group_id = str(job.get("group_id") or "")
        split = str(job.get("split") or "")
        if split not in {"train", "locked_holdout"}:
            raise ValueError(f"{name} group {group_id} has invalid split {split}")
        if group_id in group_splits and group_splits[group_id] != split:
            raise ValueError(f"{name} group {group_id} crosses split boundaries")
        group_splits[group_id] = split
    locked_count = sum(split == "locked_holdout" for split in group_splits.values())
    if locked_count != expected_locked_groups:
        raise ValueError(
            f"{name} holdout split changed: expected {expected_locked_groups} locked groups, "
            f"found {locked_count}"
        )

    terminal_failures = {"feasibility_failure", "numerical_feasibility_failure"}
    for row in rows:
        status = str(row.get("terminal_status") or "")
        group_id = str(row["group_id"])
        if status == "measured_success_gold":
            _validate_evidence(row, name=name)
        elif require_all_measured:
            raise ValueError(
                f"{name} contains non-final row; expected measured_success_gold: "
                f"{row.get('manifest_job_id')}"
            )
        elif status not in terminal_failures:
            raise ValueError(f"{name} contains non-terminal row: {row.get('manifest_job_id')}")
        if group_splits[group_id] == "locked_holdout" and status != "measured_success_gold":
            raise ValueError(f"{name} locked holdout row must be measured: {row.get('manifest_job_id')}")
    return jobs_by_id, grouped, group_splits


def _locked_snapshot(
    rows: Sequence[Mapping[str, Any]], group_splits: Mapping[str, str]
) -> dict[str, tuple[Any, ...]]:
    return {
        str(row["manifest_job_id"]): tuple(row.get(field) for field in LOCKED_FIELDS)
        for row in rows
        if group_splits[str(row["group_id"])] == "locked_holdout"
    }


def merge_gold144(
    gold128_rows: Sequence[Mapping[str, Any]],
    manifest128: Mapping[str, Any],
    targeted16_rows: Sequence[Mapping[str, Any]],
    manifest16: Mapping[str, Any],
    *,
    gold128_sha256: str,
) -> dict[str, Any]:
    if not SHA256_PATTERN.fullmatch(gold128_sha256):
        raise ValueError("gold128_sha256 must be 64 hexadecimal characters")
    jobs128, groups128, splits128 = _validate_pool(
        gold128_rows,
        manifest128,
        name="gold128",
        row_schema=GOLD128_OUTPUT_SCHEMA,
        manifest_schema=GOLD128_MANIFEST_SCHEMA,
        expected_rows=128,
        expected_groups=32,
        expected_locked_groups=6,
        require_all_measured=False,
    )
    jobs16, groups16, _ = _validate_pool(
        targeted16_rows,
        manifest16,
        name="targeted16",
        row_schema=TARGETED16_OUTPUT_SCHEMA,
        manifest_schema=TARGETED16_MANIFEST_SCHEMA,
        expected_rows=16,
        expected_groups=4,
        expected_locked_groups=0,
        require_all_measured=True,
    )

    duplicate_ids = set(jobs128) & set(jobs16)
    if duplicate_ids:
        raise ValueError(f"duplicate manifest IDs across pools: {sorted(duplicate_ids)}")
    overlap = set(groups128) & set(groups16)
    if overlap:
        raise ValueError(f"Gold128/targeted16 group overlap is forbidden: {sorted(overlap)}")

    locked_before = _locked_snapshot(gold128_rows, splits128)
    output_rows: list[dict[str, Any]] = []
    output_jobs: list[dict[str, Any]] = []
    for source_pool, rows, jobs_by_id in (
        ("gold128", gold128_rows, jobs128),
        ("targeted16", targeted16_rows, jobs16),
    ):
        for source in rows:
            manifest_id = str(source["manifest_job_id"])
            job = jobs_by_id[manifest_id]
            output_rows.append({
                **dict(source),
                "schema_version": OUTPUT_SCHEMA,
                "split": str(job["split"]),
                "source_pool": source_pool,
            })
            output_jobs.append({
                **dict(job),
                "schema_version": MANIFEST_SCHEMA,
                "source_pool": source_pool,
            })

    output_splits = {
        str(job["group_id"]): str(job["split"])
        for job in output_jobs
    }
    locked_after = _locked_snapshot(output_rows, output_splits)
    if locked_after != locked_before:
        raise ValueError("locked holdout manifest IDs, metrics, or evidence hashes changed")

    pilot_group_ids = [str(group_id) for group_id in manifest128.get("pilot_group_ids", [])]
    if len(pilot_group_ids) != 2 or not set(pilot_group_ids) <= set(groups128):
        raise ValueError("Gold128 must provide two valid pilot/reference groups")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "jobs": output_jobs,
        "pilot_group_ids": pilot_group_ids,
        "source_manifests": {
            "gold128_schema": manifest128.get("schema_version"),
            "gold128_sha256": gold128_sha256,
            "targeted16_schema": manifest16.get("schema_version"),
        },
    }
    audit = {
        "schema_version": OUTPUT_SCHEMA,
        "rows": len(output_rows),
        "groups": len(output_splits),
        "source_rows": dict(Counter(str(row["source_pool"]) for row in output_rows)),
        "split_groups": dict(Counter(output_splits.values())),
        "status_rows": dict(Counter(str(row["terminal_status"]) for row in output_rows)),
        "model_rows": dict(Counter(str(row["model"]) for row in output_rows)),
        "backend_rows": dict(Counter(str(row["dispatch_key"]) for row in output_rows)),
        "q_mode_rows": dict(Counter(str(row["q_mode"]) for row in output_rows)),
        "locked_holdout_groups": sorted(
            group_id for group_id, split in output_splits.items() if split == "locked_holdout"
        ),
    }
    expected_splits = {"train": 30, "locked_holdout": 6}
    if (
        audit["rows"] != 144
        or audit["groups"] != 36
        or audit["split_groups"] != expected_splits
    ):
        raise ValueError(f"Gold144 closure invariant failed: {audit}")
    return {"rows": output_rows, "manifest": manifest, "audit": audit}


def _csv_value(value: Any) -> Any:
    return json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value


def write_outputs(result: Mapping[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in result["rows"]]
    (output / "gold144_final.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "gold144_final.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "gold144_final.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: _csv_value(row.get(key)) for key in fieldnames}
            for row in rows
        )
    (output / "gold144_manifest.json").write_text(
        json.dumps(result["manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "gold144_audit.json").write_text(
        json.dumps(result["audit"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold128-json", type=Path, required=True)
    parser.add_argument("--manifest128-json", type=Path, required=True)
    parser.add_argument("--targeted16-json", type=Path, required=True)
    parser.add_argument("--manifest16-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gold128_sha256 = sha256_file(args.gold128_json)
    result = merge_gold144(
        json.loads(args.gold128_json.read_text(encoding="utf-8")),
        json.loads(args.manifest128_json.read_text(encoding="utf-8")),
        json.loads(args.targeted16_json.read_text(encoding="utf-8")),
        json.loads(args.manifest16_json.read_text(encoding="utf-8")),
        gold128_sha256=gold128_sha256,
    )
    result["audit"]["source_sha256"] = {
        "gold128": gold128_sha256,
        "manifest128": sha256_file(args.manifest128_json),
        "targeted16": sha256_file(args.targeted16_json),
        "manifest16": sha256_file(args.manifest16_json),
    }
    write_outputs(result, args.output_dir)
    print(json.dumps(result["audit"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
