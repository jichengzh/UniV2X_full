#!/usr/bin/env python3
"""Prepare three fresh performance plans for the frozen Stage5 Pareto selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage5.independent_validation_v1 import build_validation_request  # noqa: E402
from framework.stage5.coldstart_validation_source_v1 import (  # noqa: E402
    build_coldstart_fallback,
)
from framework.stage5.measurement_plan_v2 import build_performance_plan  # noqa: E402


def _gpus(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or any(item < 0 for item in result):
        raise ValueError("--gpus must contain non-negative GPU IDs")
    return result


def _source_evidence_paths(request: dict[str, object]) -> dict[str, Path]:
    result = {}
    for row in request["rows"]:  # type: ignore[index]
        group_id = str(row["group_id"])
        marker = Path(str(row["source_contract"]["source_done_marker"]))
        if marker.suffix != ".done":
            raise ValueError(f"invalid source marker: {group_id}")
        result[group_id] = marker.with_name(marker.stem + "_evidence.json")
    return result


def _quant_contracts(
    task_root: Path,
    request: dict[str, object],
    fallback: dict[str, Path] | None = None,
) -> dict[str, Path]:
    result = dict(fallback or {})
    for row in request["rows"]:  # type: ignore[index]
        if row["dispatch_key"] != "tvm_auto" or row["q_mode"] != "int8":
            continue
        width = "x".join(map(str, row["width"]))
        if str(row["manifest_job_id"]) in result:
            continue
        matches = sorted(
            task_root.glob(f"round_*/quant_contracts/{width}/tensor_quant_params.json")
        )
        if len(matches) != 1:
            raise ValueError(
                f"expected one frozen quant contract for {row['manifest_job_id']}, got {len(matches)}"
            )
        result[str(row["manifest_job_id"])] = matches[0]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--closure-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-artifact-root", type=Path, required=True)
    parser.add_argument("--coldstart-rows-json", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    closure = json.loads(args.closure_json.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    formal_paths = sorted(args.task_root.glob("round_*/measurement_request.json"))
    if not formal_paths:
        raise ValueError(f"task has no formal requests: {args.task_root}")
    template_rows = json.loads(formal_paths[0].read_text(encoding="utf-8"))["rows"]
    if not template_rows:
        raise ValueError("formal request has no task template row")
    formal_ids = set()
    for path in formal_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        formal_ids.update(
            str(row.get("manifest_job_id") or row.get("row_id") or "")
            for row in payload.get("rows") or []
        )
    gold_payload = json.loads(args.coldstart_rows_json.read_text(encoding="utf-8"))
    gold_rows = gold_payload if isinstance(gold_payload, list) else gold_payload["rows"]
    fallback = build_coldstart_fallback(
        repo_root=REPO_ROOT,
        output_dir=args.output_dir,
        gold_rows=gold_rows,
        selected_ids=[
            row_id
            for row_id in closure["independent_validation_ids"]
            if row_id not in formal_ids
        ],
        task_template=template_rows[0],
    )
    request = build_validation_request(
        args.task_root, closure, fallback_rows=fallback["rows"]
    )
    catalog_path = args.output_dir / "coldstart_source_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": fallback["schema_version"],
                "provenance": fallback["provenance"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    request_path = args.output_dir / "validation_request.json"
    request_path.write_text(
        json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_paths = _source_evidence_paths(request)
    source_paths.update(fallback["source_evidence_paths"])
    source_paths = {
        str(row["group_id"]): source_paths[str(row["group_id"])]
        for row in request["rows"]
    }
    quant_paths = _quant_contracts(
        args.task_root, request, fallback["quant_contract_paths"]
    )
    gpus = _gpus(args.gpus)
    for repeat_index in range(3):
        repeat_root = args.output_dir / f"repeat_{repeat_index}"
        plan = build_performance_plan(
            request,
            source_evidence_paths=source_paths,
            quant_contract_paths=quant_paths,
            remote_artifact_root=args.remote_artifact_root / f"repeat_{repeat_index}",
            gpus=gpus,
        )
        repeat_root.mkdir(parents=True, exist_ok=True)
        (repeat_root / "performance_manifest.json").write_text(
            json.dumps(plan["manifest"], ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        (repeat_root / "performance_jobs.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in plan["performance_jobs"]
            ),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "task_id": closure["task_id"],
                "configuration_count": len(request["rows"]),
                "repeat_count": 3,
                "request_json": str(request_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
