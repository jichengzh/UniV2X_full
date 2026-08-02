#!/usr/bin/env python3
"""Replace checkpoint-mismatched Gold AP labels with epoch-bound full reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "stage35_gold128_ap_label_repair_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_epoch(path: str) -> int:
    match = re.search(r"(?:bestval_at|net_epoch)(\d+)", Path(path).name)
    if not match:
        raise ValueError(f"cannot parse checkpoint epoch: {path}")
    return int(match.group(1))


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _validated_report(path: Path, expected_epoch: int) -> tuple[dict[str, float], int]:
    if not path.is_file():
        raise FileNotFoundError(f"AP repair report missing: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    nested = report.get("stage2_report") if isinstance(report.get("stage2_report"), Mapping) else report
    epoch = report.get("checkpoint_epoch")
    if epoch is None:
        epoch = nested.get("resume_epoch")
    if int(epoch or -1) != expected_epoch:
        raise ValueError(
            f"checkpoint epoch mismatch: expected={expected_epoch} measured={epoch} report={path}"
        )
    processed = int(report.get("processed_samples") or nested.get("processed_samples") or 0)
    if report.get("status") != "success" or processed < 1789:
        raise ValueError(f"AP repair report is not a successful full run: {path}")
    if int(report.get("failed_samples") or nested.get("failed_samples") or 0) != 0:
        raise ValueError(f"AP repair report contains failed samples: {path}")
    if int(report.get("fallback_samples") or nested.get("fallback_samples") or 0) != 0:
        raise ValueError(f"AP repair report contains fallback samples: {path}")
    metrics = {key: report.get(key, nested.get(key)) for key in ("ap30", "ap50", "ap70")}
    if not all(_finite(value) for value in metrics.values()):
        raise ValueError(f"AP repair report lacks finite AP30/AP50/AP70: {path}")
    return {key: float(value) for key, value in metrics.items()}, processed


def repair_ap_labels(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    repair_reports: Mapping[str, Path],
) -> dict[str, Any]:
    jobs = {
        str(job.get("job_id")): job
        for job in manifest.get("jobs", [])
        if isinstance(job, Mapping)
    }
    rows_by_id = {str(row.get("manifest_job_id")): row for row in rows}
    if not set(repair_reports) <= set(rows_by_id) or not set(repair_reports) <= set(jobs):
        raise ValueError("repair job IDs must exist in both rows and manifest")
    replacements: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    for job_id, report_path in repair_reports.items():
        source_contract = jobs[job_id].get("source_contract")
        if not isinstance(source_contract, Mapping):
            raise ValueError(f"manifest source contract missing: {job_id}")
        expected_epoch = _checkpoint_epoch(str(source_contract.get("checkpoint_path") or ""))
        metrics, processed = _validated_report(Path(report_path), expected_epoch)
        source = rows_by_id[job_id]
        replacements[job_id] = {
            **dict(source),
            **metrics,
            "ap_report_path": str(Path(report_path).resolve()),
            "ap_report_sha256": sha256_file(Path(report_path)),
            "ap_label_repair_schema": SCHEMA,
            "ap_checkpoint_epoch": expected_epoch,
        }
        audit_rows.append({
            "job_id": job_id,
            "checkpoint_epoch": expected_epoch,
            "processed_samples": processed,
            "old_ap70": source.get("ap70"),
            "new_ap70": metrics["ap70"],
            "report_path": str(Path(report_path).resolve()),
            "report_sha256": sha256_file(Path(report_path)),
        })
    output = [replacements.get(str(row.get("manifest_job_id")), dict(row)) for row in rows]
    return {
        "rows": output,
        "audit": {
            "schema_version": SCHEMA,
            "input_rows": len(rows),
            "repaired_rows": len(replacements),
            "repair_rows": audit_rows,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-json", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--repair", action="append", required=True, metavar="JOB_ID=REPORT_JSON")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-audit-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports: dict[str, Path] = {}
    for binding in args.repair:
        job_id, separator, path = binding.rpartition("=")
        if not separator or not job_id or not path:
            raise ValueError(f"invalid --repair binding: {binding}")
        reports[job_id] = Path(path)
    result = repair_ap_labels(
        json.loads(args.gold_json.read_text(encoding="utf-8")),
        json.loads(args.manifest_json.read_text(encoding="utf-8")),
        reports,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result["rows"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_audit_json.write_text(json.dumps(result["audit"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["audit"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
