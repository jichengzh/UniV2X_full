#!/usr/bin/env python3
"""Promote the verified Pyramid TRT schedule repair into a frozen bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage6.evidence_bundle_v1 import (
    apply_independent_validation,
    attach_common_hypervolume,
    build_independent_validation_index,
)
from scripts.stage6_collect_paper_evidence_v1 import _schedule_arm


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--frozen-bundle", type=Path, required=True)
    parser.add_argument("--repair-independent-audit", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    bundle = _read(args.frozen_bundle)
    if bundle.get("schema_version") != "stage6_paper_evidence_bundle_v1":
        raise ValueError("unexpected frozen bundle schema")
    for backend, arms in bundle["backends"].items():
        for arm_id, summary in arms.items():
            if backend == "trt" and arm_id == "schedule_only":
                continue
            if summary.get("independent_validation_complete") is not True:
                raise ValueError(f"frozen arm is not independently closed: {backend}:{arm_id}")

    audit = _read(args.repair_independent_audit)
    index = build_independent_validation_index([audit])
    schedule = _schedule_arm(
        args.formal_root,
        "trt",
        baseline_ap70=float(bundle["baseline"]["AP70"]),
    )
    configuration_id = "stage6|pyramid|schedule_only|trt|fp32"
    schedule["points"] = apply_independent_validation(
        schedule["points"], index, arm_id="schedule_only"
    )
    if not schedule["points"] or schedule["points"][0].get("independent_validation_passed") is not True:
        raise ValueError("repaired schedule point did not bind independent validation")
    schedule["independent_validation_complete"] = True
    schedule["independent_validation_target_ids"] = [configuration_id]

    trt_arms = dict(bundle["backends"]["trt"])
    trt_arms["schedule_only"] = schedule
    hv = attach_common_hypervolume(trt_arms)
    output = dict(bundle)
    output["backends"] = {
        **bundle["backends"],
        "trt": hv["arms"],
    }
    output["common_hv_normalization"] = {
        **bundle["common_hv_normalization"],
        "trt": hv["normalization"],
    }
    output["independent_validation_audits"] = [
        *bundle.get("independent_validation_audits", []),
        {"path": str(args.repair_independent_audit), "sha256": _sha(args.repair_independent_audit)},
    ]
    output["schedule_only_repair"] = {
        "backend": "trt",
        "reason": "epoch23_checkpoint_binding_replaces_erroneous_epoch1_ap_evaluation",
        "frozen_bundle_path": str(args.frozen_bundle),
        "frozen_bundle_sha256": _sha(args.frozen_bundle),
        "repair_independent_audit": str(args.repair_independent_audit),
        "repair_independent_audit_sha256": _sha(args.repair_independent_audit),
    }
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "output_json": str(args.output_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
