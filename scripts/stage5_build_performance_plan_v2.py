#!/usr/bin/env python3
"""Build verified performance jobs for one Stage5 v2 atomic task batch."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage5.measurement_plan_v2 import build_performance_plan


def apply_tvm_fp16_trial_policy(jobs: list[dict], max_trials: int) -> list[dict]:
    if max_trials < 0:
        raise ValueError("TVM FP16 max trials must be non-negative")
    updated = []
    for source in jobs:
        job = copy.deepcopy(source)
        if job.get("runner_key") == "tvm_fp16":
            command = list(job.get("command") or [])
            try:
                index = command.index("--max-trials")
            except ValueError as exc:
                raise ValueError("TVM FP16 job is missing --max-trials") from exc
            command[index + 1] = str(max_trials)
            job = {**job, "command": command, "stage6_tvm_fp16_max_trials": max_trials}
        updated.append(job)
    return updated


def _parse_gpus(value: str) -> list[int]:
    gpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not gpus or any(gpu < 0 for gpu in gpus):
        raise ValueError("--gpus must contain non-negative GPU ids")
    return gpus


def _evidence_paths(request: dict) -> dict[str, Path]:
    paths = {}
    for row in request.get("rows") or []:
        group_id = str(row.get("group_id") or "")
        marker = str((row.get("source_contract") or {}).get("source_done_marker") or "")
        if not group_id or not marker.endswith(".done"):
            raise ValueError(f"invalid source marker for {group_id or '<empty>'}")
        evidence = Path(marker[:-5] + "_evidence.json")
        previous = paths.setdefault(group_id, evidence)
        if previous != evidence:
            raise ValueError(f"inconsistent source evidence path for {group_id}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--remote-artifact-root", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quant-contract-root", type=Path, required=True)
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--tvm-fp16-max-trials", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request = json.loads(args.request_json.read_text(encoding="utf-8"))
    result = build_performance_plan(
        request,
        source_evidence_paths=_evidence_paths(request),
        quant_contract_paths={
            str(row["manifest_job_id"]): args.quant_contract_root
            / "x".join(map(str, row["width"]))
            / "tensor_quant_params.json"
            for row in request.get("rows") or []
            if row.get("dispatch_key") == "tvm_auto" and row.get("q_mode") == "int8"
        },
        remote_artifact_root=args.remote_artifact_root,
        gpus=_parse_gpus(args.gpus),
    )
    result = {
        **result,
        "manifest": {
            **result["manifest"],
            "tvm_fp16_max_trials": args.tvm_fp16_max_trials,
        },
        "performance_jobs": apply_tvm_fp16_trial_policy(
            result["performance_jobs"], args.tvm_fp16_max_trials
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "performance_manifest.json"
    jobs = args.output_dir / "performance_jobs.jsonl"
    manifest.write_text(
        json.dumps(result["manifest"], ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    jobs.write_text(
        "".join(json.dumps(job, ensure_ascii=False, sort_keys=True) + "\n" for job in result["performance_jobs"])
    )
    print(json.dumps({"manifest": str(manifest), "jobs": str(jobs), "row_count": 4}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
