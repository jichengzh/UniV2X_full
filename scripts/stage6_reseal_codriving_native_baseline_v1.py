#!/usr/bin/env python3
"""Reseal a CoDriving native baseline after the legacy file-SHA bug."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    args = parser.parse_args()

    baseline = _read(args.baseline_json)
    if baseline.get("schema_version") != "stage6_native_fp32_baseline_v3":
        raise ValueError("unexpected native baseline schema")
    ap_path = Path(str(baseline.get("ap_report_path") or ""))
    checkpoint_path = Path(str(baseline.get("checkpoint_path") or ""))
    if not ap_path.is_file() or not checkpoint_path.is_file():
        raise ValueError("baseline evidence file is missing")
    report = _read(ap_path)
    if report.get("status") != "success" or int(report.get("processed_samples") or 0) != 1789:
        raise ValueError("native AP report is not a successful full-1789 result")
    for key in ("ap30", "ap50", "ap70"):
        left = float(baseline[key])
        right = float(report[key])
        if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"baseline/report metric drift: {key}")

    old_ap_sha = str(baseline.get("ap_report_sha256") or "")
    old_checkpoint_sha = str(baseline.get("checkpoint_sha256") or "")
    baseline = {
        **baseline,
        "ap_report_sha256": _sha(ap_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "sha_contract": "standard_file_bytes_sha256_v1",
    }
    args.baseline_json.write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = {
        "schema_version": "stage6_native_baseline_sha_reseal_audit_v1",
        "status": "passed",
        "reason": "legacy_sha256_path_included_filename_for_single_file",
        "metrics_unchanged": True,
        "processed_samples": 1789,
        "baseline_json": str(args.baseline_json),
        "baseline_json_sha256": _sha(args.baseline_json),
        "ap_report_path": str(ap_path),
        "old_ap_report_sha256": old_ap_sha,
        "ap_report_sha256": baseline["ap_report_sha256"],
        "old_checkpoint_sha256": old_checkpoint_sha,
        "checkpoint_sha256": baseline["checkpoint_sha256"],
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
