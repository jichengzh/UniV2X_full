#!/usr/bin/env python3
"""Collect SHA-audited CoDriving Stage6 arm evidence into paper-table schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.stage6_collect_paper_evidence_v1 as base  # noqa: E402
import framework.stage6.evidence_bundle_v1 as evidence_bundle  # noqa: E402
from framework.stage6.evidence_bundle_v1 import (  # noqa: E402
    apply_independent_validation,
    attach_common_hypervolume,
    build_independent_validation_index,
)


base.BACKEND_TASK = {"tvm": "S5-COD-TVM", "trt": "S5-COD-TRT"}
evidence_bundle.EXPECTED_MODEL = "codriving"
evidence_bundle.BACKEND_CONTEXT = {
    "trt": ("S5-COD-TRT", "trt_engine", "h800-trt-probe-conditioned-v3"),
    "tvm": ("S5-COD-TVM", "tvm_auto", "h800-tvm-probe-conditioned-v3"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--joint-root", type=Path, required=True)
    parser.add_argument("--coldstart-rows-json", type=Path, required=True)
    parser.add_argument("--independent-audit", type=Path, action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = base._baseline(args.formal_root)
    coldstart = base._rows(base._read(args.coldstart_rows_json))
    audits = [base._read(path) for path in args.independent_audit]
    validation_index = build_independent_validation_index(audits) if audits else {}
    backends = {}
    normalizations = {}
    for backend in ("tvm", "trt"):
        raw_arms = {
            "compression_only": base._actual_arm(args.formal_root, backend, "compression_only"),
            "schedule_only": _schedule_arm(args.formal_root, backend, baseline_ap70=float(baseline["AP70"])),
            "compress_then_tune": base._actual_arm(args.formal_root, backend, "compress_then_tune"),
            "tune_then_compress": base._tune_then_compress_arm(args.formal_root, backend),
            "joint_shcosearch": base._joint_arm(backend=backend, joint_root=args.joint_root, coldstart_rows=coldstart),
        }
        hv = attach_common_hypervolume(raw_arms)
        backends[backend] = base._apply_validation(hv["arms"], validation_index, float(baseline["AP70"]))
        normalizations[backend] = hv["normalization"]
    bundle = {
        "schema_version": "stage6_paper_evidence_bundle_v1",
        "target_model": "codriving",
        "baseline": baseline,
        "backends": backends,
        "common_hv_normalization": normalizations,
        "independent_validation_audits": [
            {"path": str(path), "sha256": base._sha(path)}
            for path in args.independent_audit
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_json": str(args.output_json),
        "arm_statuses": {
            backend: {arm: summary["status"] for arm, summary in arms.items()}
            for backend, arms in backends.items()
        },
    }, sort_keys=True))
    return 0


def _schedule_arm(
    formal_root: Path, backend: str, *, baseline_ap70: float
) -> dict[str, object]:
    arm = base._schedule_arm(formal_root, backend, baseline_ap70=baseline_ap70)
    for point in arm.get("points") or []:
        point["manifest_job_id"] = f"stage6|codriving|schedule_only|{backend}|fp32"
    return arm


if __name__ == "__main__":
    raise SystemExit(main())
