#!/usr/bin/env python3
"""Prepare arm-scoped Stage6 independent performance reruns."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from framework.stage5.measurement_plan_v2 import build_performance_plan  # noqa: E402
from framework.stage6.evidence_bundle_v1 import (  # noqa: E402
    build_independent_validation_index,
)
from framework.stage6.independent_validation_v1 import (  # noqa: E402
    PIPELINE_IDS,
    build_scoped_validation_request,
)
from scripts.stage5_build_performance_plan_v2 import (  # noqa: E402
    apply_tvm_fp16_trial_policy,
)


TASKS = {"tvm": "S5-PYR-TVM", "trt": "S5-PYR-TRT"}
SEARCH_ARMS = ("compression_only", "compress_then_tune")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _gpus(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or any(item < 0 for item in result):
        raise ValueError("--gpus must contain non-negative GPU IDs")
    return result


def _request_rows(
    formal_root: Path, backend: str, arm_id: str
) -> dict[str, tuple[dict[str, Any], Path]]:
    task = TASKS[backend]
    paths = sorted(
        (formal_root / backend / arm_id).glob(
            f"formal_batch_*/{task}/round_*/measurement_request.json"
        )
    )
    result: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in paths:
        for source in _read(path).get("rows") or []:
            row = dict(source)
            row_id = str(row.get("manifest_job_id") or "")
            previous = result.get(row_id)
            if previous is not None and previous[0] != row:
                raise ValueError(f"conflicting Stage6 request row: {row_id}")
            result[row_id] = (row, path)
    return result


def _source_evidence_path(row: Mapping[str, Any]) -> Path:
    marker = Path(str((row.get("source_contract") or {}).get("source_done_marker") or ""))
    if marker.suffix != ".done":
        raise ValueError(f"invalid source marker: {row.get('manifest_job_id')}")
    path = marker.with_name(marker.stem + "_evidence.json")
    if not path.is_file():
        raise ValueError(f"source evidence is missing: {path}")
    return path


def _quant_paths(
    rows_and_requests: list[tuple[dict[str, Any], Path]],
) -> dict[str, Path]:
    result = {}
    for row, request_path in rows_and_requests:
        if row.get("dispatch_key") != "tvm_auto" or row.get("q_mode") != "int8":
            continue
        width = "x".join(map(str, row["width"]))
        path = request_path.parent / "quant_contracts" / width / "tensor_quant_params.json"
        if not path.is_file():
            raise ValueError(f"frozen quant contract is missing: {path}")
        result[str(row["manifest_job_id"])] = path
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jobs(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prepare_arm(
    *,
    formal_root: Path,
    output_root: Path,
    backend: str,
    arm_id: str,
    summary: Mapping[str, Any],
    validation_index: Mapping[tuple[str, str], Mapping[str, Any]],
    gpus: list[int],
) -> dict[str, Any] | None:
    if summary.get("status") != "complete":
        return None
    targets = [
        target
        for target in summary.get("independent_validation_target_ids") or []
        if (arm_id, str(target)) not in validation_index
    ]
    if not targets:
        return None
    catalog = _request_rows(formal_root, backend, arm_id)
    missing = sorted(set(map(str, targets)) - set(catalog))
    if missing:
        raise ValueError(f"validation targets not found in arm requests: {missing}")
    selected = [catalog[str(target)] for target in targets]
    rows = [row for row, _ in selected]
    request = build_scoped_validation_request(rows, arm_id=arm_id, backend=backend)
    arm_root = output_root / backend / arm_id
    _write_json(arm_root / "validation_request.json", request)
    points = {
        str(point["manifest_job_id"]): point for point in summary.get("points") or []
    }
    _write_json(
        arm_root / "reference_points.json",
        [points[str(target)] for target in targets],
    )
    source_paths = {
        str(row["group_id"]): _source_evidence_path(row) for row in rows
    }
    quant_paths = _quant_paths(selected)
    max_trials = 0 if arm_id == "compression_only" else 64
    for repeat in range(3):
        repeat_root = arm_root / f"repeat_{repeat}"
        plan = build_performance_plan(
            request,
            source_evidence_paths=source_paths,
            quant_contract_paths=quant_paths,
            remote_artifact_root=repeat_root / "performance_execution",
            gpus=gpus,
        )
        jobs = apply_tvm_fp16_trial_policy(plan["performance_jobs"], max_trials)
        manifest = {
            **plan["manifest"],
            "stage6_arm_id": arm_id,
            "stage6_pipeline_id": PIPELINE_IDS[(backend, arm_id)],
            "tvm_fp16_max_trials": max_trials if backend == "tvm" else None,
        }
        _write_json(repeat_root / "performance_manifest.json", manifest)
        repeat_root.mkdir(parents=True, exist_ok=True)
        _write_jobs(repeat_root / "performance_jobs.jsonl", jobs)
    return {
        "backend": backend,
        "arm_id": arm_id,
        "pipeline_id": PIPELINE_IDS[(backend, arm_id)],
        "configuration_ids": list(map(str, targets)),
        "configuration_count": len(targets),
        "repeat_count": 3,
        "root": str(arm_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--evidence-bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--independent-audit", type=Path, action="append", default=[])
    parser.add_argument("--gpus", default="3,4,5,6,7")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = _read(args.evidence_bundle)
    if bundle.get("schema_version") != "stage6_paper_evidence_bundle_v1":
        raise ValueError("unexpected paper evidence bundle schema")
    audits = [_read(path) for path in args.independent_audit]
    validation_index = build_independent_validation_index(audits) if audits else {}
    plans = []
    for backend in ("tvm", "trt"):
        for arm_id in SEARCH_ARMS:
            plan = _prepare_arm(
                formal_root=args.formal_root,
                output_root=args.output_root,
                backend=backend,
                arm_id=arm_id,
                summary=bundle["backends"][backend][arm_id],
                validation_index=validation_index,
                gpus=_gpus(args.gpus),
            )
            if plan is not None:
                plans.append(plan)
    manifest = {
        "schema_version": "stage6_independent_validation_plan_v1",
        "plans": plans,
        "plan_count": len(plans),
    }
    _write_json(args.output_root / "stage6_independent_validation_plan_v1.json", manifest)
    for index, plan in enumerate(plans):
        _write_json(
            args.output_root / f"stage6_independent_validation_plan_{index:02d}.json",
            {
                "schema_version": "stage6_independent_validation_plan_v1",
                "plans": [plan],
                "plan_count": 1,
            },
        )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
