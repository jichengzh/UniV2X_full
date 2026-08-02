#!/usr/bin/env python3
"""Prepare independent reruns for selected CoDriving joint-search winners."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from framework.stage5.coldstart_validation_source_v1 import (  # noqa: E402
    build_coldstart_fallback,
)
from framework.stage5.measurement_plan_v2 import build_performance_plan  # noqa: E402
from framework.stage6.independent_validation_v1 import (  # noqa: E402
    PIPELINE_IDS,
    build_scoped_validation_request,
)
from scripts.stage5_build_performance_plan_v2 import (  # noqa: E402
    apply_tvm_fp16_trial_policy,
)


TASKS = {"tvm": "S5-COD-TVM", "trt": "S5-COD-TRT"}
ARM_ID = "joint_shcosearch"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jobs(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _gpus(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or any(item < 0 for item in result):
        raise ValueError("--gpus must contain non-negative GPU IDs")
    return result


def _request_catalog(task_root: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    result = {}
    for path in sorted(task_root.glob("round_*/measurement_request.json")):
        for source in _read(path).get("rows") or []:
            row = dict(source)
            row_id = str(row.get("manifest_job_id") or "")
            if row_id:
                result[row_id] = (row, path)
    return result


def _source_evidence(row: Mapping[str, Any]) -> Path:
    marker = Path(str((row.get("source_contract") or {}).get("source_done_marker") or ""))
    path = marker.with_name(marker.stem + "_evidence.json")
    if marker.suffix != ".done" or not path.is_file():
        raise ValueError(f"source evidence missing: {row.get('manifest_job_id')}")
    return path


def _existing_quant_contract(task_root: Path, row: Mapping[str, Any]) -> Path:
    width = "x".join(map(str, row["width"]))
    matches = sorted(task_root.glob(f"round_*/quant_contracts/{width}/tensor_quant_params.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one quant contract for {row['manifest_job_id']}, got {len(matches)}")
    return matches[0]


def _prepare_backend(
    *,
    backend: str,
    joint_root: Path,
    evidence: Mapping[str, Any],
    gold_rows: list[Mapping[str, Any]],
    output_root: Path,
    gpus: list[int],
) -> dict[str, Any]:
    task = TASKS[backend]
    task_root = joint_root / task
    summary = evidence["backends"][backend][ARM_ID]
    if summary.get("status") != "complete":
        raise ValueError(f"joint search is incomplete: {backend}")
    targets = list(summary.get("independent_validation_target_ids") or [])
    if not targets:
        raise ValueError(f"joint search has no validation targets: {backend}")
    catalog = _request_catalog(task_root)
    template = next(iter(catalog.values()))[0]
    arm_root = output_root / backend / ARM_ID
    missing = [target for target in targets if target not in catalog]
    fallback = build_coldstart_fallback(
        repo_root=REPO,
        output_dir=arm_root / "coldstart_fallback",
        gold_rows=gold_rows,
        selected_ids=missing,
        task_template=template,
    )
    fallback_rows = dict(fallback["rows"])
    selected = [catalog[target][0] if target in catalog else fallback_rows[target] for target in targets]
    request = build_scoped_validation_request(selected, arm_id=ARM_ID, backend=backend)
    _write(arm_root / "validation_request.json", request)
    source_paths = {
        str(row["group_id"]): (
            fallback["source_evidence_paths"][str(row["group_id"])]
            if str(row["manifest_job_id"]) in fallback_rows
            else _source_evidence(row)
        )
        for row in selected
    }
    points = {str(point["manifest_job_id"]): point for point in summary.get("points") or []}
    references = []
    for target, row in zip(targets, selected):
        source = points[target]
        references.append({
            **source,
            "evidence_files": {
                **dict(source.get("evidence_files") or {}),
                "source": str(source_paths[str(row["group_id"])]),
            },
        })
    _write(arm_root / "reference_points.json", references)
    quant_paths = dict(fallback["quant_contract_paths"])
    for row in selected:
        row_id = str(row["manifest_job_id"])
        if row.get("dispatch_key") == "tvm_auto" and row.get("q_mode") == "int8" and row_id not in quant_paths:
            quant_paths[row_id] = _existing_quant_contract(task_root, row)

    for repeat in range(3):
        repeat_root = arm_root / f"repeat_{repeat}"
        plan = build_performance_plan(
            request,
            source_evidence_paths=source_paths,
            quant_contract_paths=quant_paths,
            remote_artifact_root=repeat_root / "performance_execution",
            gpus=gpus,
        )
        jobs = apply_tvm_fp16_trial_policy(plan["performance_jobs"], 64)
        manifest = {
            **plan["manifest"],
            "stage6_arm_id": ARM_ID,
            "stage6_pipeline_id": PIPELINE_IDS[(backend, ARM_ID)],
            "tvm_fp16_max_trials": 64 if backend == "tvm" else None,
        }
        repeat_root.mkdir(parents=True, exist_ok=True)
        _write(repeat_root / "performance_manifest.json", manifest)
        _write_jobs(repeat_root / "performance_jobs.jsonl", jobs)
    return {
        "backend": backend,
        "arm_id": ARM_ID,
        "pipeline_id": PIPELINE_IDS[(backend, ARM_ID)],
        "configuration_ids": targets,
        "configuration_count": len(targets),
        "repeat_count": 3,
        "root": str(arm_root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-root", type=Path, required=True)
    parser.add_argument("--evidence-bundle", type=Path, required=True)
    parser.add_argument("--coldstart-rows-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", default="3")
    args = parser.parse_args()
    payload = _read(args.coldstart_rows_json)
    gold_rows = payload if isinstance(payload, list) else payload["rows"]
    evidence = _read(args.evidence_bundle)
    gpus = _gpus(args.gpus)
    plans = [
        _prepare_backend(
            backend=backend,
            joint_root=args.joint_root,
            evidence=evidence,
            gold_rows=gold_rows,
            output_root=args.output_root,
            gpus=gpus,
        )
        for backend in ("tvm", "trt")
    ]
    manifest = {
        "schema_version": "stage6_independent_validation_plan_v1",
        "plans": plans,
        "plan_count": len(plans),
    }
    _write(args.output_root / "stage6_joint_independent_validation_plan_v1.json", manifest)
    for index, plan in enumerate(plans):
        _write(
            args.output_root / f"stage6_joint_independent_validation_plan_{index:02d}.json",
            {"schema_version": manifest["schema_version"], "plans": [plan], "plan_count": 1},
        )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
