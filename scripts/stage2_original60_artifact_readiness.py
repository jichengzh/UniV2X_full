#!/usr/bin/env python3
"""Validate original60 artifact readiness on the H800 artifact filesystem."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


SCHEMA = "stage2_original60_artifact_readiness_v1"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    item = Path(path)
    if not item.exists():
        return []
    return [
        json.loads(line)
        for line in item.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe(value: object) -> str:
    return str(value).replace("/", "-").replace(":", "-").replace(" ", "_")


def _label(candidate: dict[str, Any]) -> str:
    label = str(candidate.get("label") or "")
    if label:
        return _safe(label)
    candidate_id = str(candidate.get("candidate_id") or "unknown")
    return _safe(candidate_id.split(":")[-1])


def json_readable(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return False
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        pass
    try:
        for line in text.splitlines():
            if line.strip():
                json.loads(line)
        return True
    except json.JSONDecodeError:
        return False


def collect_job_state(paths: list[str]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for pattern in paths:
        for path in glob.glob(pattern):
            for row in read_jsonl(path):
                label = str(row.get("label") or "")
                status = str(row.get("status") or "")
                if label:
                    statuses[label] = status
    return statuses


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    candidates = read_jsonl(args.candidate_queue)
    if len(candidates) != 60:
        raise SystemExit(f"expected exactly 60 original candidates, got {len(candidates)}")

    root = Path(args.artifact_root)
    job_status = collect_job_state(args.job_state_glob or [])
    active_quarantine_rows = read_jsonl(args.quarantine_file) if args.quarantine_file else []
    active_quarantine = [
        row
        for row in active_quarantine_rows
        if str(row.get("status") or row.get("artifact_status") or "").lower()
        not in {"resolved", "ready", "succeeded"}
    ]

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        label = _label(candidate)
        onnx_path = root / "models" / f"{label}_backbone.onnx"
        work_dir = root / "workdirs" / label
        workload_path = work_dir / "database_workload.json"
        tuning_record_path = work_dir / "database_tuning_record.json"
        checks = {
            "onnx_exists": onnx_path.is_file() and onnx_path.stat().st_size > 0,
            "workdir_exists": work_dir.is_dir(),
            "database_workload_readable": json_readable(workload_path),
            "database_tuning_record_readable": json_readable(tuning_record_path),
        }
        ready = all(checks.values())
        rows.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "label": label,
                "width": candidate.get("width"),
                "status": "ready" if ready else "missing",
                "worker_status": job_status.get(label, ""),
                "checks": checks,
                "onnx_path": str(onnx_path),
                "tvm_work_dir": str(work_dir),
                "database_workload_path": str(workload_path),
                "database_tuning_record_path": str(tuning_record_path),
            }
        )

    ready_count = sum(1 for row in rows if row["status"] == "ready")
    missing_rows = [row for row in rows if row["status"] != "ready"]
    summary = {
        "schema": SCHEMA,
        "candidate_count": len(candidates),
        "original60_artifact_ready_count": ready_count,
        "missing_artifact_count": len(missing_rows),
        "active_quarantine_count": len(active_quarantine),
        "is_complete": ready_count == 60 and not missing_rows and not active_quarantine,
        "artifact_root": str(root),
        "rows": rows,
    }
    return summary


def write_markdown(path: str | Path, summary: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage2 original60 artifact readiness",
        "",
        f"- candidate_count: {summary['candidate_count']}",
        f"- original60_artifact_ready_count: {summary['original60_artifact_ready_count']}",
        f"- missing_artifact_count: {summary['missing_artifact_count']}",
        f"- active_quarantine_count: {summary['active_quarantine_count']}",
        f"- is_complete: {summary['is_complete']}",
        "",
        "| label | width | status | worker_status | missing checks |",
        "|---|---:|---|---|---|",
    ]
    for row in summary["rows"]:
        missing = ",".join(k for k, v in row["checks"].items() if not v)
        lines.append(
            "| {label} | {width} | {status} | {worker_status} | {missing} |".format(
                label=row["label"],
                width="x".join(str(v) for v in (row.get("width") or [])),
                status=row["status"],
                worker_status=row.get("worker_status") or "",
                missing=missing,
            )
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-queue", required=True)
    parser.add_argument("--artifact-root", default="/exdata/jichengzhi/s2_tvm")
    parser.add_argument("--job-state-glob", action="append", default=[])
    parser.add_argument("--quarantine-file", default="")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize(args)
    write_json(args.json_out, summary)
    write_markdown(args.md_out, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, sort_keys=True))
    if args.require_complete and not summary["is_complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
