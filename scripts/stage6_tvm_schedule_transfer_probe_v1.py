#!/usr/bin/env python3
"""Probe whether a frozen base-shape TVM database fully transfers to a compressed graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2_h800_run_measurement_job import configure_tvm_env, load_relax_module  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--base-work-dir", type=Path, required=True)
    parser.add_argument("--base-width", default="64,128,256")
    parser.add_argument("--compressed-width", default="48,96,192")
    parser.add_argument("--intended-q-mode", choices=["fp16", "int8"], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_tvm_env(str(args.gpu))
    import tvm
    from tvm import relax
    import tvm.s_tir.tensor_intrin.cuda  # noqa: F401

    device = tvm.cuda(0)
    target = tvm.target.Target.from_device(device)
    module, _, _ = load_relax_module(args.onnx)
    pipeline = tvm.transform.Sequential(
        [
            relax.transform.LegalizeOps(),
            relax.transform.AnnotateTIROpPattern(),
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
        ]
    )
    with target, tvm.transform.PassContext(opt_level=3):
        fused = pipeline(module)
        transferred = relax.transform.MetaScheduleApplyDatabase(
            work_dir=str(args.base_work_dir)
        )(fused)
    total_primfuncs = 0
    changed_primfuncs = 0
    changed_names = []
    for global_var, before in fused.functions.items():
        if type(before).__name__ != "PrimFunc":
            continue
        total_primfuncs += 1
        after = transferred[global_var]
        if not tvm.ir.structural_equal(before, after, map_free_vars=True):
            changed_primfuncs += 1
            changed_names.append(global_var.name_hint)
    ratio = changed_primfuncs / total_primfuncs if total_primfuncs else 0.0
    full_transfer = total_primfuncs > 0 and changed_primfuncs == total_primfuncs
    workload = args.base_work_dir / "database_workload.json"
    records = args.base_work_dir / "database_tuning_record.json"
    payload = {
        "schema_version": "stage6_tvm_base_policy_transfer_probe_v1",
        "base_width": [int(value) for value in args.base_width.split(",")],
        "compressed_width": [int(value) for value in args.compressed_width.split(",")],
        "intended_q_mode": args.intended_q_mode,
        "q_dispatch_reached": False,
        "blocked_stage": "frozen_base_policy_transfer",
        "onnx_path": str(args.onnx),
        "onnx_sha256": sha256_file(args.onnx),
        "base_work_dir": str(args.base_work_dir),
        "database_workload_sha256": sha256_file(workload),
        "database_tuning_record_sha256": sha256_file(records),
        "total_primfuncs": total_primfuncs,
        "base_policy_applied_primfuncs": changed_primfuncs,
        "base_policy_coverage": ratio,
        "changed_primfunc_names": changed_names,
        "full_transfer": full_transfer,
        "compressed_shape_retuned": False,
        "fallback_used": False,
        "terminal_status": "transferred_success" if full_transfer else "feasibility_failure",
        "failure_reason": None if full_transfer else "base_schedule_database_not_applicable_to_all_compressed_primfuncs",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
