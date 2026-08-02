#!/usr/bin/env python3
"""Initialize and audit the formal Pyramid Stage6 six-arm execution state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ARM_IDS = (
    "original_default",
    "compression_only",
    "schedule_only",
    "compress_then_tune",
    "tune_then_compress",
    "joint_shcosearch",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slot_status(output_root: Path, arm: str, backend: str) -> str:
    terminal = output_root / backend / arm / "terminal_summary.json"
    if terminal.is_file():
        payload = json.loads(terminal.read_text(encoding="utf-8"))
        return str(payload.get("status") or "terminal_summary_present")
    if arm == "joint_shcosearch":
        return "evidence_bound_pending_validation"
    if arm == "original_default":
        report = output_root / "native_fp32_ap/full_ap_eval_report.json"
        bound = output_root / "stage6_native_fp32_baseline_with_ap_v2.json"
        if report.is_file() and bound.is_file():
            payload = json.loads(bound.read_text(encoding="utf-8"))
            if payload.get("ap_evidence_status") == "full_h800_backend_execution_bound":
                return "complete"
        return "running"
    return "pending"


def build_runner_state(plan: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    if plan.get("schema_version") != "stage6_formal_execution_plan_v1" or plan.get("passed") is not True:
        raise ValueError("formal execution plan is not validated")
    if set(plan.get("arms") or {}) != set(ARM_IDS):
        raise ValueError("formal execution plan does not cover the frozen six arms")
    slots = {"original_default|pytorch_eager": _slot_status(output_root, "original_default", "pytorch_eager")}
    for arm in ARM_IDS[1:]:
        for backend in ("tvm", "trt"):
            slots[f"{arm}|{backend}"] = _slot_status(output_root, arm, backend)
    complete = {"complete", "complete_failure", "paper_ready"}
    return {
        "schema_version": "stage6_formal_runner_state_v1",
        "plan_sha256": str(plan["plan_sha256"]),
        "formal_stage6_started": True,
        "slots": slots,
        "complete_slot_count": sum(status in complete for status in slots.values()),
        "slot_count": len(slots),
        "paper_table_ready": all(status in complete for status in slots.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if _sha(args.plan) == str(plan.get("plan_sha256")):
        raise ValueError("plan_sha256 must digest the plan payload, not the serialized envelope")
    state = build_runner_state(plan, args.output_root)
    output = args.state_out or args.output_root / "stage6_formal_runner_state_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(state, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
