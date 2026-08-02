#!/usr/bin/env python3
"""Build isolated H800 latency/energy repeat jobs for four Gold128 anchor classes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "stage35_gold128_repeat_job_v1"
ANCHOR_GROUPS = {
    "base": ("pyramid|32x32x64", "codriving|48x96x192"),
    "small_channel": ("pyramid|16x32x64", "codriving|16x32x64"),
    "alignment_trap": ("pyramid|48x64x128", "codriving|48x64x128"),
    "large_model": ("pyramid|64x128x256", "codriving|64x128x256"),
}
EXPECTED_ARMS = {"tvm_fp16", "tvm_int8", "trt_fp16", "trt_int8"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _replace_root(value: Any, old_root: str, new_root: str) -> Any:
    if isinstance(value, str):
        return value.replace(old_root, new_root)
    if isinstance(value, list):
        return [_replace_root(item, old_root, new_root) for item in value]
    if isinstance(value, dict):
        return {key: _replace_root(item, old_root, new_root) for key, item in value.items()}
    return value


def _replace_or_append_flag(command: Sequence[str], flag: str, value: str) -> list[str]:
    updated = list(command)
    if flag not in updated:
        return [*updated, flag, value]
    index = updated.index(flag)
    if index + 1 >= len(updated):
        raise ValueError(f"command flag lacks value: {flag}")
    return [*updated[: index + 1], value, *updated[index + 2 :]]


def _bind_repaired_tvm_int8_contract(
    job: Mapping[str, Any], baseline_result_json: str
) -> dict[str, Any]:
    result_path = Path(baseline_result_json)
    if result_path.name != "route_b_int8_auto_decomp_result.json":
        raise ValueError(f"TVM-INT8 Gold baseline is not an automatic decomposition result: {result_path}")
    if len(result_path.parents) < 3 or result_path.parents[1].name != "build":
        raise ValueError(f"TVM-INT8 Gold baseline lacks repaired build contract: {result_path}")
    quant_contract = result_path.parents[2] / "tensor_quant_params.json"
    label = result_path.parent.name
    if not label.endswith("_scaleaware"):
        raise ValueError(f"TVM-INT8 Gold baseline is not scale-aware: {result_path}")
    command = list(job.get("command") or [])
    for flag, value in (
        ("--label", label),
        ("--tensor-quant-params-json", str(quant_contract)),
        ("--warmup", "20"),
        ("--number", "20"),
        ("--repeat", "5"),
    ):
        command = _replace_or_append_flag(command, flag, value)
    return {
        **dict(job),
        "command": command,
        "measurement_contract": "repaired_scaleaware_auto_tvm_int8_v1",
        "tensor_quant_params_json": str(quant_contract),
    }


def build_repeat_plan(
    source_jobs: Sequence[Mapping[str, Any]],
    gold_rows: Sequence[Mapping[str, Any]],
    *,
    output_root: str,
    gpu: int,
) -> list[dict[str, Any]]:
    selected_groups = {group for groups in ANCHOR_GROUPS.values() for group in groups}
    by_group: dict[str, list[Mapping[str, Any]]] = {}
    seen_manifest_ids: set[str] = set()
    for row in source_jobs:
        group_id = str(row.get("group_id"))
        if group_id not in selected_groups:
            continue
        manifest_id = str(row.get("manifest_job_id"))
        if manifest_id in seen_manifest_ids:
            raise ValueError(f"duplicate source job for {manifest_id}")
        seen_manifest_ids.add(manifest_id)
        by_group.setdefault(group_id, []).append(row)
    baseline_by_id = {str(row.get("manifest_job_id")): row for row in gold_rows}

    output: list[dict[str, Any]] = []
    for category, groups in ANCHOR_GROUPS.items():
        for group_id in groups:
            group_rows = by_group.get(group_id, [])
            arms = {str(row.get("runner_key")) for row in group_rows}
            if len(group_rows) != 4 or arms != EXPECTED_ARMS:
                raise ValueError(f"anchor {category}/{group_id} lacks a complete four-arm source plan")
            for source in sorted(group_rows, key=lambda row: str(row["runner_key"])):
                manifest_id = str(source["manifest_job_id"])
                baseline = baseline_by_id.get(manifest_id)
                if baseline is None or not baseline.get("performance_result_json"):
                    raise ValueError(f"missing Gold128 baseline result for {manifest_id}")
                old_root = str(source.get("remote_artifact_root") or "")
                if not old_root:
                    raise ValueError(f"source job lacks remote_artifact_root: {manifest_id}")
                new_group_root = f"{output_root}/{category}/{group_id}"
                runner = str(source["runner_key"])
                rewritten = _replace_root(dict(source), old_root, new_group_root)
                if runner == "tvm_int8":
                    rewritten = _bind_repaired_tvm_int8_contract(
                        rewritten, str(baseline["performance_result_json"])
                    )
                output.append({
                    **rewritten,
                    "schema_version": SCHEMA,
                    "job_id": f"repeat|{category}|{group_id}|{runner}",
                    "baseline_job_id": str(source["job_id"]),
                    "baseline_manifest_job_id": manifest_id,
                    "baseline_result_json": str(baseline["performance_result_json"]),
                    "repeat_category": category,
                    "repeat_round": 1,
                    "assigned_gpu": gpu,
                    "gpu_pool": str(gpu),
                    "remote_artifact_root": output_root,
                    "terminal_status": "pending",
                })
    if len(output) != 32 or len({row["job_id"] for row in output}) != 32:
        raise ValueError("repeat plan must contain 32 unique jobs")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jobs-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--gold-json", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--only-runner-key", choices=sorted(EXPECTED_ARMS))
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-plan-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    jobs = [row for path in args.source_jobs_jsonl for row in read_jsonl(path)]
    gold = json.loads(args.gold_json.read_text(encoding="utf-8"))
    output = build_repeat_plan(jobs, gold, output_root=args.output_root, gpu=args.gpu)
    if args.only_runner_key:
        output = [row for row in output if row["runner_key"] == args.only_runner_key]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output), encoding="utf-8"
    )
    summary = {
        "schema_version": SCHEMA,
        "jobs": len(output),
        "groups": len({row["group_id"] for row in output}),
        "categories": {category: sum(row["repeat_category"] == category for row in output) for category in ANCHOR_GROUPS},
        "only_runner_key": args.only_runner_key,
        "gpu": args.gpu,
        "output_root": args.output_root,
    }
    args.output_plan_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
