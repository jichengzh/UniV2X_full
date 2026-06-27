#!/usr/bin/env python3
"""Plan original60 artifact build queues for Stage2 H800 LUT production."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = "/exdata/jichengzhi/s2_tvm"
SCHEMA = "stage2_original60_artifact_build_job_v1"
MANIFEST_SCHEMA = "stage2_original60_artifact_manifest_row_v1"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    item = Path(path)
    if not item.exists():
        return []
    return [
        json.loads(line)
        for line in item.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _safe(value: object) -> str:
    return str(value).replace("/", "-").replace(":", "-").replace(" ", "_")


def _width(candidate: dict[str, Any]) -> list[int]:
    width = candidate.get("width")
    if not isinstance(width, list) or len(width) != 3:
        raise ValueError(f"candidate has invalid width: {candidate.get('candidate_id')}")
    return [int(item) for item in width]


def _label(candidate: dict[str, Any]) -> str:
    label = str(candidate.get("label") or "")
    if label:
        return _safe(label)
    candidate_id = str(candidate.get("candidate_id") or "unknown")
    return _safe(candidate_id.split(":")[-1])


def build_rows(
    candidates: list[dict[str, Any]],
    *,
    artifact_root: str,
    local_onnx_dir: str,
    created_at: str,
    gpus: list[int],
    max_trials: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        label = _label(candidate)
        width = _width(candidate)
        root = Path(artifact_root)
        tvm_work_dir = root / "workdirs" / label
        onnx_path = root / "models" / f"{label}_backbone.onnx"
        row = {
            "schema": SCHEMA,
            "job_id": f"artifact_original60:{label}",
            "candidate_id": str(candidate["candidate_id"]),
            "config_id": str(candidate.get("config_id") or candidate["candidate_id"]),
            "label": label,
            "width": width,
            "quant_policy": str(candidate.get("quant_policy") or "fp16"),
            "optimized_scope": str(candidate.get("optimized_scope") or "backbone_only"),
            "dense_stage": str(candidate.get("dense_stage") or "backbone"),
            "artifact_root": str(root),
            "local_onnx_path": str(Path(local_onnx_dir) / f"{label}_backbone.onnx"),
            "onnx_path": str(onnx_path),
            "tvm_work_dir": str(tvm_work_dir),
            "database_workload_path": str(tvm_work_dir / "database_workload.json"),
            "database_tuning_record_path": str(tvm_work_dir / "database_tuning_record.json"),
            "gpu": int(gpus[index % len(gpus)]),
            "max_trials": int(max_trials),
            "created_at": created_at,
            "source_candidate": candidate,
        }
        rows.append(row)
    return rows


def parse_gpus(value: str) -> list[int]:
    gpus = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not gpus:
        raise ValueError("at least one GPU must be specified")
    return gpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-queue", required=True)
    parser.add_argument("--out-queue", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--shard-dir")
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--local-onnx-dir",
        default="multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/artifacts/original60_onnx",
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--max-trials", type=int, default=32)
    parser.add_argument("--created-at", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = read_jsonl(args.candidate_queue)
    if len(candidates) != 60:
        raise SystemExit(f"expected exactly 60 original candidates, got {len(candidates)}")
    gpus = parse_gpus(args.gpus)
    rows = build_rows(
        candidates,
        artifact_root=args.artifact_root,
        local_onnx_dir=args.local_onnx_dir,
        created_at=args.created_at,
        gpus=gpus,
        max_trials=args.max_trials,
    )
    write_jsonl(args.out_queue, rows)
    manifest_rows = [
        {
            "schema": MANIFEST_SCHEMA,
            "candidate_id": row["candidate_id"],
            "label": row["label"],
            "width": row["width"],
            "onnx_path": row["onnx_path"],
            "tvm_work_dir": row["tvm_work_dir"],
            "database_workload_path": row["database_workload_path"],
            "database_tuning_record_path": row["database_tuning_record_path"],
            "gpu": row["gpu"],
            "max_trials": row["max_trials"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]
    write_jsonl(args.out_manifest, manifest_rows)
    if args.shard_dir:
        shard_dir = Path(args.shard_dir)
        for gpu in gpus:
            write_jsonl(
                shard_dir / f"artifact_build_queue_original60_gpu{gpu}.jsonl",
                [row for row in rows if int(row["gpu"]) == gpu],
            )
    print(
        json.dumps(
            {
                "schema": "stage2_original60_artifact_build_plan_summary_v1",
                "candidate_count": len(candidates),
                "job_count": len(rows),
                "gpus": gpus,
                "max_trials": args.max_trials,
                "out_queue": args.out_queue,
                "out_manifest": args.out_manifest,
                "shard_dir": args.shard_dir,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
