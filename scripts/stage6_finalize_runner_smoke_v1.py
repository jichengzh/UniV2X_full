#!/usr/bin/env python3
"""Assemble Stage6 protocol and H800 runner smoke evidence into one gate artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage6.protocol_smoke_v1 import (  # noqa: E402
    lock_compress_then_tune,
    record_reverse_transfer_attempts,
    select_hardware_blind_batch,
)


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def protocol_smoke() -> dict[str, Any]:
    candidates = [
        {
            "candidate_id": f"blind-{index:02d}",
            "width": [16 + 8 * (index % 4), 32 + 16 * (index // 4), 64],
            "q_mode": "fp16" if index % 2 else "int8",
            "parameter_count": 1_000_000 + index * 10_000,
            "flops": 10_000_000_000 + index * 100_000_000,
            "ap_surrogate": 0.50 + index * 0.01,
            "latency_ms": float(index),
            "energy_j": float(12 - index),
        }
        for index in range(12)
    ]
    first = select_hardware_blind_batch(candidates, batch_size=4)
    perturbed = [
        {**row, "latency_ms": 10_000 - index, "energy_j": 10_000 + index}
        for index, row in enumerate(candidates)
    ]
    second = select_hardware_blind_batch(perturbed, batch_size=4)
    blind_ids = [row["candidate_id"] for row in first]
    blind_perturbed_ids = [row["candidate_id"] for row in second]
    forward = lock_compress_then_tune(
        [{"candidate_id": row["candidate_id"], "screen_rank": index} for index, row in enumerate(candidates)]
    )
    return {
        "hardware_blind": {
            "selected_ids": blind_ids,
            "selected_ids_after_backend_label_perturbation": blind_perturbed_ids,
            "passed": blind_ids == blind_perturbed_ids,
        },
        "compress_then_tune": {
            **forward,
            "passed": forward["screen_count"] == 12 and forward["locked_count"] == 4,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--trt", type=Path, required=True)
    parser.add_argument("--tvm", type=Path, required=True)
    parser.add_argument("--reverse-tvm", type=Path, required=True)
    parser.add_argument("--reverse-trt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    native, trt, tvm, reverse_tvm, reverse_trt = map(
        read, (args.native, args.trt, args.tvm, args.reverse_tvm, args.reverse_trt)
    )
    protocols = protocol_smoke()
    reverse_accounting = record_reverse_transfer_attempts(
        [
            {
                "candidate_id": "pyramid|48x96x192|backend=tvm",
                "applicable": reverse_tvm.get("full_transfer") is True,
                "reason": reverse_tvm.get("failure_reason"),
                "evidence": str(args.reverse_tvm),
                "fallback_used": reverse_tvm.get("fallback_used") is True,
                "compressed_shape_retuned": reverse_tvm.get("compressed_shape_retuned") is True,
            },
            {
                "candidate_id": "pyramid|48x96x192|backend=trt",
                "applicable": reverse_trt.get("full_transfer") is True,
                "reason": reverse_trt.get("failure_reason"),
                "evidence": str(args.reverse_trt),
                "fallback_used": reverse_trt.get("fallback_used") is True,
                "compressed_shape_retuned": reverse_trt.get("compressed_shape_retuned") is True,
            },
        ]
    )
    checks = {
        "native_fp32_h800_exact_scope": (
            native.get("hardware") == "NVIDIA H800"
            and native.get("scope") == "pyramid_multiscale_backbone"
            and native.get("input_shape") == [2, 64, 128, 256]
            and int(native.get("independent_repeat_count") or 0) >= 3
        ),
        "trt_fp32_schedule_runner": (
            trt.get("precision") == "fp32"
            and trt.get("build_success") is True
            and trt.get("numerical_finite") is True
            and bool(trt.get("artifact_sha256"))
            and trt.get("warmup") == 20
            and trt.get("iters") == 300
            and trt.get("repeat") == 5
            and trt.get("n_lat_samples") == 1500
            and float(trt.get("energy_secs") or 0) >= 5.0
        ),
        "tvm_fp32_schedule_runner": (
            tvm.get("precision") == "fp32"
            and tvm.get("build_success") is True
            and float(tvm.get("lat_tuned_ms") or 0) > 0
            and float(tvm.get("energy_j") or 0) > 0
        ),
        "compression_only_no_backend_label_leakage": protocols["hardware_blind"]["passed"],
        "compress_then_tune_12_plus_4_lock": protocols["compress_then_tune"]["passed"],
        "tune_then_compress_failure_accounted_without_fallback": (
            reverse_accounting["attempt_count"] == 2
            and reverse_accounting["fallback_count"] == 0
            and reverse_accounting["retune_count"] == 0
            and int(reverse_tvm.get("total_primfuncs") or 0) > 0
            and reverse_tvm.get("terminal_status") in {"transferred_success", "feasibility_failure"}
            and reverse_trt.get("error_on_timing_cache_miss") is True
            and reverse_trt.get("terminal_status") in {"transferred_success", "feasibility_failure"}
        ),
    }
    payload = {
        "schema_version": "stage6_runner_smoke_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "protocol_smoke": protocols,
        "reverse_transfer_accounting": reverse_accounting,
        "evidence_paths": {
            "native": str(args.native),
            "trt": str(args.trt),
            "tvm": str(args.tvm),
            "reverse_tvm": str(args.reverse_tvm),
            "reverse_trt": str(args.reverse_trt),
        },
        "formal_stage6_rows_emitted": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
