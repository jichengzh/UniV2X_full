#!/usr/bin/env python3
"""Generate a full60 FP16 rewritten artifact latency/energy measurement queue."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def read_summary(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"summary rows not found in {path}")
    return [row for row in rows if row.get("precision") == "fp16"]


def parse_width(value: Any) -> str:
    if isinstance(value, str):
        return value.replace("x", ",")
    if isinstance(value, list):
        return ",".join(str(int(item)) for item in value)
    raise ValueError(f"unsupported width: {value!r}")


def label_from_run_dir(path: Path) -> str | None:
    match = re.match(r"(.+)_gpu[0-9]+$", path.name)
    return match.group(1) if match else None


def build_artifact_index(root: Path) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for artifact in root.glob("*_gpu*/rewrite_raw/*_full_engine_group_conv_rewrite/rewritten_full_engine.so"):
        run_dir = artifact.parents[2]
        label = label_from_run_dir(run_dir)
        if not label:
            continue
        reports = list((run_dir / "rewrite_exports").glob(f"fp16_{label}_full_engine_group_conv_rewrite_latest.json"))
        if not reports:
            reports = list((run_dir / "rewrite_exports").glob("*full_engine_group_conv_rewrite_latest.json"))
        if not reports:
            continue
        index[label] = {"artifact": str(artifact), "rewrite_report": str(reports[0])}
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--gpus", default="0,6")
    args = parser.parse_args()

    rows = read_summary(args.summary_json)
    artifact_index = build_artifact_index(args.artifact_root)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")

    queue: list[dict[str, Any]] = []
    missing: list[str] = []
    for idx, row in enumerate(sorted(rows, key=lambda item: str(item.get("label")))):
        label = str(row["label"])
        artifact = artifact_index.get(label)
        if not artifact:
            missing.append(label)
            continue
        width = parse_width(row.get("width"))
        queue.append(
            {
                "label": label,
                "width": width,
                "gpu": gpus[idx % len(gpus)],
                "artifact": artifact["artifact"],
                "rewrite_report": artifact["rewrite_report"],
                "config_id": f"fp16_rewritten_tensorcore_full60_{label}",
                "route": "apshape_tensorcore_rewritten_full60",
            }
        )
    if missing:
        raise ValueError(f"missing rewritten artifacts for labels: {missing}")
    if len(queue) != 60:
        raise ValueError(f"expected 60 queue rows, got {len(queue)}")

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for item in queue:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"event": "queue_generated", "rows": len(queue), "out": str(args.out_jsonl)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
