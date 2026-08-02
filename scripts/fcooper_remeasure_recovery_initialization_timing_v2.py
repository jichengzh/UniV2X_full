#!/usr/bin/env python3
"""Remeasure missing F-Cooper recovery-initialization wall times."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


SUCCESS = "measured_success_gold"
BASE_WIDTH = (64, 128, 256, 128, 256)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def missing_widths(payloads: Sequence[Mapping[str, Any]]) -> list[tuple[int, ...]]:
    widths: set[tuple[int, ...]] = set()
    for payload in payloads:
        for row in payload.get("rows") or []:
            timings = row.get("phase_timings_seconds") or {}
            width = tuple(int(value) for value in row.get("width") or ())
            if (
                row.get("terminal_status") == SUCCESS
                and width
                and width != BASE_WIDTH
                and not isinstance(
                    timings.get("recovery_initialization_seconds"),
                    (int, float),
                )
            ):
                widths.add(width)
    return sorted(widths)


def run_one(args: argparse.Namespace, width: tuple[int, ...]) -> dict[str, Any]:
    tag = "x".join(map(str, width))
    output_dir = args.work_root / tag
    if output_dir.exists():
        shutil.rmtree(output_dir)
    command = [
        str(args.python),
        str(args.code_root / "scripts/fcooper_recovery_v2.py"),
        "--source-config",
        str(args.source_config),
        "--source-checkpoint",
        str(args.source_checkpoint),
        "--width",
        ",".join(map(str, width)),
        "--recovery-contract",
        str(args.recovery_contract),
        "--output-dir",
        str(output_dir),
    ]
    started = time.monotonic()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": f"{args.code_root}:{args.heal_root}",
            "CUDA_VISIBLE_DEVICES": "",
        },
    )
    elapsed = time.monotonic() - started
    log_path = args.log_root / f"{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    report_path = output_dir / "recovery_initialization_report.json"
    if completed.returncode != 0 or not report_path.is_file():
        raise RuntimeError(f"initialization timing remeasurement failed for {tag}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "ready_for_recovery_training"
        or tuple(report.get("width") or ()) != width
        or not math.isfinite(elapsed)
        or elapsed <= 0.0
    ):
        raise ValueError(f"invalid initialization timing evidence for {tag}")
    return {
        "width": list(width),
        "status": "success",
        "elapsed_seconds": elapsed,
        "timing_kind": "independent_monotonic_remeasurement",
        "command": command,
        "command_sha256": hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode()
        ).hexdigest(),
        "report_path": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
        "log_path": str(log_path.resolve()),
        "log_sha256": sha256_file(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-json", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--heal-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--recovery-contract", type=Path, required=True)
    args = parser.parse_args()

    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.feedback_json
    ]
    widths = missing_widths(payloads)
    entries = [run_one(args, width) for width in widths]
    summary = {
        "schema_version": "fcooper_recovery_initialization_timing_v2",
        "passed": len(entries) == len(widths),
        "entry_count": len(entries),
        "entries": entries,
        "source_config_sha256": sha256_file(args.source_config),
        "source_checkpoint_sha256": sha256_file(args.source_checkpoint),
        "recovery_contract_sha256": sha256_file(args.recovery_contract),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": summary["passed"], "entries": len(entries)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
