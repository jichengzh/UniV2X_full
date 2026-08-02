#!/usr/bin/env python3
"""Promote the verified Pyramid TVM default/schedule repair into a bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage6.evidence_bundle_v1 import attach_common_hypervolume
from scripts.stage6_collect_paper_evidence_v1 import (
    _schedule_arm,
    _tvm_default_baseline,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--frozen-bundle", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    bundle = _read(args.frozen_bundle)
    if bundle.get("schema_version") != "stage6_paper_evidence_bundle_v1":
        raise ValueError("unexpected frozen bundle schema")
    for backend, arms in bundle["backends"].items():
        for arm_id, summary in arms.items():
            if backend == "tvm" and arm_id == "schedule_only":
                continue
            if summary.get("status") == "complete":
                if summary.get("independent_validation_complete") is not True:
                    raise ValueError(
                        f"frozen arm is not independently closed: {backend}:{arm_id}"
                    )
            elif summary.get("status") == "complete_failure":
                if summary.get("failure_evidence_sha_verified") is not True:
                    raise ValueError(
                        f"frozen failure is not verified: {backend}:{arm_id}"
                    )
            else:
                raise ValueError(f"frozen arm is nonterminal: {backend}:{arm_id}")

    tvm_baseline = _tvm_default_baseline(args.formal_root)
    if tvm_baseline is None:
        raise ValueError("verified TVM default repair is unavailable")
    schedule = _schedule_arm(
        args.formal_root,
        "tvm",
        baseline_ap70=float(tvm_baseline["AP70"]),
    )
    if (
        schedule.get("status") != "complete"
        or schedule.get("independent_validation_complete") is not True
        or len(schedule.get("points") or []) != 1
    ):
        raise ValueError("TVM schedule repair did not close")

    tvm_arms = {**bundle["backends"]["tvm"], "schedule_only": schedule}
    hv = attach_common_hypervolume(tvm_arms)
    output = {
        **bundle,
        "backends": {**bundle["backends"], "tvm": hv["arms"]},
        "common_hv_normalization": {
            **bundle["common_hv_normalization"],
            "tvm": hv["normalization"],
        },
        "backend_baselines": {
            **(bundle.get("backend_baselines") or {}),
            "trt": bundle["baseline"],
            "tvm": tvm_baseline,
        },
        "tvm_default_schedule_repair": {
            "reason": "replace_native_cudnn_row_with_tvm_zero_trial_and_bind_epoch23_schedule_ap",
            "frozen_bundle_path": str(args.frozen_bundle),
            "frozen_bundle_sha256": _sha(args.frozen_bundle),
            "repair_audit": tvm_baseline["evidence_path"],
            "repair_audit_sha256": tvm_baseline["evidence_sha256"],
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "passed", "output_json": str(args.output_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
