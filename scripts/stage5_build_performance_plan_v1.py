#!/usr/bin/env python3
"""Build verified Stage5 performance jobs for a real feedback round."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage5.measurement_plan_v1 import build_performance_plan


def _parse_gpus(value: str) -> list[int]:
    gpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not gpus or any(gpu < 0 for gpu in gpus):
        raise ValueError("--gpus must contain non-negative GPU ids")
    return gpus


def _evidence_paths(request: dict) -> dict[str, Path]:
    paths: dict[str, Path] = {}
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
    parser.add_argument("--gpus", default="6,7")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request = json.loads(args.request_json.read_text(encoding="utf-8"))
    result = build_performance_plan(
        request,
        source_evidence_paths=_evidence_paths(request),
        remote_artifact_root=args.remote_artifact_root,
        gpus=_parse_gpus(args.gpus),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "performance_manifest.json"
    jobs_path = args.output_dir / "performance_jobs.jsonl"
    manifest_path.write_text(
        json.dumps(result["manifest"], ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    jobs_path.write_text(
        "".join(
            json.dumps(job, ensure_ascii=False, sort_keys=True) + "\n"
            for job in result["performance_jobs"]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema_version": "stage5_performance_plan_outputs_v1",
                "manifest": str(manifest_path.resolve()),
                "jobs": str(jobs_path.resolve()),
                "group_count": result["manifest"]["group_count"],
                "row_count": result["manifest"]["row_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
