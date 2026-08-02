#!/usr/bin/env python3
"""Freeze the executable Pyramid Stage6 candidate and arm plans."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage6.formal_plan_v1 import (  # noqa: E402
    build_formal_plan,
    fit_neutral_ap_surrogate,
)
from framework.stage5.single_target_search_v2 import (  # noqa: E402
    SearchTask,
    build_measurement_request,
)


def _rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"expected candidate rows: {path}")
    return [dict(row) for row in rows]


def _key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    genome = list(row["genome"])
    return int(genome[0]), int(genome[1]), int(genome[2]), str(genome[3])


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = ROOT / "results/stage5_single_target_search_v2_20260718"
    parser.add_argument(
        "--tvm-candidates",
        type=Path,
        default=default_root / "S5-PYR-TVM/round_00/predicted_candidates.json",
    )
    parser.add_argument(
        "--trt-candidates",
        type=Path,
        default=default_root / "S5-PYR-TRT/round_00/predicted_candidates.json",
    )
    parser.add_argument("--expected-pool-size", type=int, default=100)
    coldstart = ROOT / "results/stage35_gold144_targeted_supplement_v2_20260714/final_gold176_v1"
    parser.add_argument("--coldstart-rows", type=Path, default=coldstart / "gold176_final.json")
    parser.add_argument("--coldstart-graphs", type=Path, default=coldstart / "graph_features.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profiles-json",
        type=Path,
        default=ROOT / "results/s1_profile_final_v3_20260711/capability_profiles_v3.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backend_rows = {"tvm": _rows(args.tvm_candidates), "trt": _rows(args.trt_candidates)}
    neutral_ap = fit_neutral_ap_surrogate(
        _rows(args.coldstart_rows),
        _rows(args.coldstart_graphs),
        backend_rows["tvm"],
    )
    plan = build_formal_plan(
        backend_rows["tvm"],
        backend_rows["trt"],
        expected_pool_size=args.expected_pool_size,
        neutral_ap_by_genome=neutral_ap,
    )
    _write(args.output_dir / "stage6_formal_execution_plan_v1.json", plan)
    for backend, rows in backend_rows.items():
        indexed = {_key(row): row for row in rows}
        profiles_payload = json.loads(args.profiles_json.read_text(encoding="utf-8"))
        profiles = (
            profiles_payload.get("capability_profiles")
            if isinstance(profiles_payload, dict)
            else profiles_payload
        )
        dispatch_key = "tvm_auto" if backend == "tvm" else "trt_engine"
        profile = next(row for row in profiles if row["dispatch_key"] == dispatch_key)
        task = SearchTask(f"S5-PYR-{backend.upper()}", "pyramid", "h800", profile)
        for arm in ("compression_only", "compress_then_tune", "tune_then_compress"):
            selected = plan["backend_plans"][backend][arm]
            payload = {
                "schema_version": "stage6_formal_arm_candidate_plan_v1",
                "plan_sha256": plan["plan_sha256"],
                "backend": backend,
                "arm_id": arm,
                "candidate_count": len(selected),
                "rows": [indexed[tuple(genome)] for genome in selected],
            }
            _write(args.output_dir / backend / f"{arm}_candidate_plan.json", payload)
            if arm in {"compression_only", "compress_then_tune"}:
                for batch_index in range(0, len(selected), 4):
                    batch_rows = [indexed[tuple(genome)] for genome in selected[batch_index : batch_index + 4]]
                    request = build_measurement_request(
                        task=task,
                        selected_rows=batch_rows,
                        round_index=batch_index // 4,
                    )
                    _write(
                        args.output_dir
                        / backend
                        / arm
                        / f"batch_{batch_index // 4:02d}"
                        / "measurement_request.json",
                        request,
                    )
    runner = ROOT / "scripts/stage6_formal_runner_v1.py"
    runner_plan = {
        "schema_version": "stage6_formal_runner_plan_v1",
        "experiment_id": "stage6-pyramid-h800-six-arm-v1",
        "target_model": "pyramid",
        "hardware_id": "h800",
        "backends": ["tvm", "trt"],
        "arm_ids": [
            "original_default",
            "compression_only",
            "schedule_only",
            "compress_then_tune",
            "tune_then_compress",
            "joint_shcosearch",
        ],
        "formal_execution_plan": str((args.output_dir / "stage6_formal_execution_plan_v1.json").resolve()),
        "formal_execution_plan_sha256": _sha(args.output_dir / "stage6_formal_execution_plan_v1.json"),
        "formal_runner_path": str(runner.resolve()),
        "formal_runner_sha256": _sha(runner),
        "output_root": str(args.output_dir.resolve()),
        "budget_contract_bound": True,
    }
    _write(args.output_dir / "stage6_formal_runner_plan_v1.json", runner_plan)
    print(json.dumps({"plan_sha256": plan["plan_sha256"], "passed": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
