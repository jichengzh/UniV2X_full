#!/usr/bin/env python3
"""Build two H800 performance batches for the frozen Gold32 supplement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import stage3_gold96_performance_batches_v3 as gold96


SOURCE_SCHEMA = "stage35_gold32_supplement_manifest_v1"
SCHEMA_VERSION = "stage35_gold32_performance_batch_plan_v1"


def _require_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError(f"expected {SOURCE_SCHEMA} manifest")
    rows = manifest.get("jobs")
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError("Gold32 supplement manifest must contain 32 rows")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for source in rows:
        row = dict(source)
        grouped.setdefault(str(row["group_id"]), []).append(row)
    if len(grouped) != 8 or any(len(group_rows) != 4 for group_rows in grouped.values()):
        raise ValueError("Gold32 supplement must contain eight complete four-row groups")
    return [dict(row) for row in rows]


def _build_job(
    row: Mapping[str, Any],
    *,
    batch_index: int,
    row_index: int,
    remote_artifact_root: str | Path,
    gpus: Sequence[int],
) -> dict[str, Any]:
    source_contract = row.get("source_contract")
    if not isinstance(source_contract, Mapping):
        raise ValueError(f"row missing source_contract: {row.get('job_id')}")
    onnx_path = str(source_contract.get("onnx_path") or "")
    calibration_root = str(source_contract.get("calibration_root") or "")
    if not onnx_path or not calibration_root:
        raise ValueError(f"incomplete source_contract: {row.get('job_id')}")
    runner_key = gold96._runner_key(row)
    assigned_gpu = int(gpus[row_index % len(gpus)])
    command = gold96._build_command(
        row=row,
        batch_index=batch_index,
        runner_key=runner_key,
        onnx_path=onnx_path,
        calibration_root=calibration_root,
        assigned_gpu=assigned_gpu,
        remote_artifact_root=remote_artifact_root,
    )
    output_dir = gold96._job_output_dir(remote_artifact_root, batch_index, row, runner_key)
    return {
        "schema_version": "stage35_gold32_performance_job_v1",
        "job_id": f"{row['group_id']}|{runner_key}",
        "manifest_job_id": str(row["job_id"]),
        "group_id": str(row["group_id"]),
        "model": str(row["model"]),
        "width_key": gold96._width_key(row),
        "q_mode": str(row["q_mode"]),
        "runner_key": runner_key,
        "dispatch_key": str(row["dispatch_key"]),
        "split": str(row["split"]),
        "onnx_path": onnx_path,
        "calibration_root": calibration_root,
        "source_contract": dict(source_contract),
        "command": command,
        "assigned_gpu": assigned_gpu,
        "gpu_pool": ",".join(str(int(gpu)) for gpu in gpus),
        "remote_artifact_root": str(remote_artifact_root),
        "expected_result_json": str(output_dir / "result.json"),
        "max_attempts": 2,
        "terminal_status": "pending",
    }


def build_all_batches(
    manifest: Mapping[str, Any],
    *,
    remote_artifact_root: str | Path,
    gpus: Sequence[int],
) -> list[dict[str, Any]]:
    rows = _require_manifest(manifest)
    if not gpus:
        raise ValueError("gpus must not be empty")
    group_ids = sorted({str(row["group_id"]) for row in rows})
    group_batches = (group_ids[:4], group_ids[4:])
    batches: list[dict[str, Any]] = []
    for batch_index, batch_group_ids in enumerate(group_batches, start=1):
        selected = [row for row in rows if str(row["group_id"]) in set(batch_group_ids)]
        selected.sort(key=lambda row: (str(row["group_id"]), gold96._runner_rank(row)))
        jobs = [
            _build_job(
                row,
                batch_index=batch_index,
                row_index=row_index,
                remote_artifact_root=remote_artifact_root,
                gpus=gpus,
            )
            for row_index, row in enumerate(selected)
        ]
        batches.append(
            {
                "schema_version": SCHEMA_VERSION,
                "batch_index": batch_index,
                "group_ids": list(batch_group_ids),
                "group_count": len(batch_group_ids),
                "manifest_row_count": len(selected),
                "manifest_rows": selected,
                "jobs": jobs,
            }
        )
    return batches


def _parse_gpus(value: str) -> list[int]:
    gpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not gpus or any(gpu < 0 for gpu in gpus):
        raise ValueError("--gpus must contain non-negative GPU ids")
    return gpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--remote-artifact-root", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="7")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    batches = build_all_batches(
        manifest,
        remote_artifact_root=args.remote_artifact_root,
        gpus=_parse_gpus(args.gpus),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, str]] = []
    for batch in batches:
        batch_index = int(batch["batch_index"])
        plan_path = args.output_dir / f"gold32_performance_batch_{batch_index:02d}_plan.json"
        jobs_path = args.output_dir / f"gold32_performance_batch_{batch_index:02d}_jobs.jsonl"
        plan_path.write_text(json.dumps(batch, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        jobs_path.write_text(
            "".join(json.dumps(job, ensure_ascii=False, sort_keys=True) + "\n" for job in batch["jobs"]),
            encoding="utf-8",
        )
        outputs.append({"plan_json": str(plan_path.resolve()), "jobs_jsonl": str(jobs_path.resolve())})
    print(json.dumps({"schema_version": SCHEMA_VERSION, "batches": outputs}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
