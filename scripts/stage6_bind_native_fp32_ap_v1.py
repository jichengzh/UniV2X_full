#!/usr/bin/env python3
"""Bind a full native-FP32 AP report to the measured Stage6 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_ap_report(baseline_path: Path, report_path: Path) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if baseline.get("schema_version") != "stage6_native_fp32_baseline_v1":
        raise ValueError("unexpected native baseline schema")
    if report.get("status") != "success" or report.get("precision") != "fp32":
        raise ValueError("AP report is not a successful FP32 evaluation")
    if report.get("precision_mode") != "model_float32" or int(report.get("num_samples", 0)) != 1789:
        raise ValueError("AP report is not the frozen full validation evaluation")
    if Path(str(report.get("ckpt_path") or "")) != Path(str(baseline.get("checkpoint_path") or "")):
        raise ValueError("checkpoint path drift between performance and AP")
    checkpoint = Path(str(baseline["checkpoint_path"]))
    if not checkpoint.is_file() or _sha(checkpoint) != baseline.get("checkpoint_sha256"):
        raise ValueError("baseline checkpoint SHA verification failed")
    if report.get("checkpoint_sha256") != baseline.get("checkpoint_sha256"):
        raise ValueError("AP report checkpoint SHA drift")
    metrics = {}
    for key in ("ap30", "ap50", "ap70"):
        value = report.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"invalid {key}")
        metrics[key] = float(value)
    return {
        **baseline,
        **metrics,
        "ap_num_samples": 1789,
        "ap_report_path": str(report_path.resolve()),
        "ap_report_sha256": _sha(report_path),
        "ap_checkpoint_sha256": str(baseline["checkpoint_sha256"]),
        "ap_evidence_status": "full_h800_backend_execution_bound",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--ap-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = bind_ap_report(args.baseline, args.ap_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "ap70": result["ap70"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
