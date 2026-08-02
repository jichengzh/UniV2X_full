#!/usr/bin/env python3
"""Merge frozen Gold144 with the measured 32-row targeted supplement."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage35_gold144_merge_v1 import (
    SHA256_PATTERN,
    _csv_value,
    _locked_snapshot,
    _validate_pool,
    sha256_file,
)


GOLD144_OUTPUT_SCHEMA = "stage35_gold144_final_v1"
GOLD144_MANIFEST_SCHEMA = "stage35_gold144_manifest_v1"
TARGETED32_OUTPUT_SCHEMA = "stage35_targeted32_final_v1"
TARGETED32_MANIFEST_SCHEMA = "stage35_gold176_targeted_supplement_manifest_v1"
OUTPUT_SCHEMA = "stage35_gold176_final_v1"
MANIFEST_SCHEMA = "stage35_gold176_manifest_v1"


def merge_gold176(
    gold144_rows: Sequence[Mapping[str, Any]],
    manifest144: Mapping[str, Any],
    targeted32_rows: Sequence[Mapping[str, Any]],
    manifest32: Mapping[str, Any],
    *,
    gold144_sha256: str,
) -> dict[str, Any]:
    if not SHA256_PATTERN.fullmatch(gold144_sha256):
        raise ValueError("gold144_sha256 must be 64 hexadecimal characters")
    jobs144, groups144, splits144 = _validate_pool(
        gold144_rows,
        manifest144,
        name="gold144",
        row_schema=GOLD144_OUTPUT_SCHEMA,
        manifest_schema=GOLD144_MANIFEST_SCHEMA,
        expected_rows=144,
        expected_groups=36,
        expected_locked_groups=6,
        require_all_measured=False,
    )
    jobs32, groups32, _ = _validate_pool(
        targeted32_rows,
        manifest32,
        name="targeted32",
        row_schema=TARGETED32_OUTPUT_SCHEMA,
        manifest_schema=TARGETED32_MANIFEST_SCHEMA,
        expected_rows=32,
        expected_groups=8,
        expected_locked_groups=0,
        require_all_measured=True,
    )
    if set(jobs144) & set(jobs32):
        raise ValueError("duplicate manifest IDs across Gold144 and targeted32")
    if set(groups144) & set(groups32):
        raise ValueError("Gold144/targeted32 group overlap is forbidden")

    locked_before = _locked_snapshot(gold144_rows, splits144)
    rows: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for source_pool, source_rows, source_jobs in (
        ("gold144", gold144_rows, jobs144),
        ("targeted32", targeted32_rows, jobs32),
    ):
        for source in source_rows:
            manifest_id = str(source["manifest_job_id"])
            job = source_jobs[manifest_id]
            rows.append(
                {
                    **dict(source),
                    "schema_version": OUTPUT_SCHEMA,
                    "split": str(job["split"]),
                    "source_pool": source_pool,
                }
            )
            jobs.append(
                {
                    **dict(job),
                    "schema_version": MANIFEST_SCHEMA,
                    "source_pool": source_pool,
                }
            )

    splits = {str(job["group_id"]): str(job["split"]) for job in jobs}
    if _locked_snapshot(rows, splits) != locked_before:
        raise ValueError("locked holdout metrics or evidence hashes changed")
    pilot_group_ids = [str(value) for value in manifest144.get("pilot_group_ids", [])]
    if len(pilot_group_ids) != 2 or not set(pilot_group_ids) <= set(groups144):
        raise ValueError("Gold144 must preserve two pilot/reference groups")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "jobs": jobs,
        "pilot_group_ids": pilot_group_ids,
        "source_manifests": {
            "gold144_schema": manifest144.get("schema_version"),
            "gold144_sha256": gold144_sha256,
            "targeted32_schema": manifest32.get("schema_version"),
        },
    }
    audit = {
        "schema_version": OUTPUT_SCHEMA,
        "rows": len(rows),
        "groups": len(splits),
        "source_rows": dict(Counter(str(row["source_pool"]) for row in rows)),
        "split_groups": dict(Counter(splits.values())),
        "status_rows": dict(Counter(str(row["terminal_status"]) for row in rows)),
        "model_rows": dict(Counter(str(row["model"]) for row in rows)),
        "backend_rows": dict(Counter(str(row["dispatch_key"]) for row in rows)),
        "q_mode_rows": dict(Counter(str(row["q_mode"]) for row in rows)),
        "locked_holdout_groups": sorted(group for group, split in splits.items() if split == "locked_holdout"),
    }
    if audit["rows"] != 176 or audit["groups"] != 44 or audit["split_groups"] != {"train": 38, "locked_holdout": 6}:
        raise ValueError(f"Gold176 closure invariant failed: {audit}")
    return {"rows": rows, "manifest": manifest, "audit": audit}


def write_outputs(result: Mapping[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in result["rows"]]
    (output / "gold176_final.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    (output / "gold176_final.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "gold176_final.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _csv_value(row.get(key)) for key in fields} for row in rows)
    (output / "gold176_manifest.json").write_text(json.dumps(result["manifest"], indent=2, sort_keys=True) + "\n")
    (output / "gold176_audit.json").write_text(json.dumps(result["audit"], indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold144-json", type=Path, required=True)
    parser.add_argument("--manifest144-json", type=Path, required=True)
    parser.add_argument("--targeted32-json", type=Path, required=True)
    parser.add_argument("--manifest32-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = merge_gold176(
        json.loads(args.gold144_json.read_text()),
        json.loads(args.manifest144_json.read_text()),
        json.loads(args.targeted32_json.read_text()),
        json.loads(args.manifest32_json.read_text()),
        gold144_sha256=sha256_file(args.gold144_json),
    )
    result["audit"]["source_sha256"] = {
        "gold144": sha256_file(args.gold144_json),
        "manifest144": sha256_file(args.manifest144_json),
        "targeted32": sha256_file(args.targeted32_json),
        "manifest32": sha256_file(args.manifest32_json),
    }
    write_outputs(result, args.output_dir)
    print(json.dumps(result["audit"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
