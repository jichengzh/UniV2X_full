#!/usr/bin/env python3
"""Run a diagnostic normalized smoke for capability-conditioned precision selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from framework.stage2.canonical_search_v3 import build_capability_profile  # noqa: E402
from framework.stage2.cost_model_bundle_v3 import (  # noqa: E402
    backend_blind_bundle,
    fit_model_bundle,
    predict_rows,
)
from framework.stage2.search_loop_v3 import (  # noqa: E402
    best_q_by_profile,
    unconditional_int8_regret,
)
DEFAULT_ROOT = REPO / "results/canonical_search_v3_20260711"
DEFAULT_PROFILES = DEFAULT_ROOT / "capability_profiles_v3.json"
DEFAULT_DECISION = REPO / "results/codriving_source_conv_auto_20260711/final_backend_decision.json"
DEFAULT_OUT = DEFAULT_ROOT / "architecture_smoke_v3.json"

MINIMAL_DECISION_FIXTURE = {
    "tvm_top2_region": {"fp16_median_ms": 0.1674816, "int8_median_ms": 0.21646235},
    "trt_32x64x128_backbone": {"fp16_p50_ms": 0.4243839979, "int8_p50_ms": 0.2965279967},
}

MINIMAL_PROFILES_FIXTURE = [
    build_capability_profile(
        capability_profile_id="h800-tvm-auto-v3",
        hardware_target="h800",
        compiler_fingerprint="a" * 64,
        dispatch_key="tvm_auto",
        features={
            "supports_fp16_tensorcore": 1.0,
            "supports_int8_tensorcore": 1.0,
            "int8_precision_propagation_ratio": 0.25,
            "qdq_fold_ratio": 0.1,
            "reformat_rate": 0.4,
        },
    ),
    build_capability_profile(
        capability_profile_id="h800-trt-v3",
        hardware_target="h800",
        compiler_fingerprint="b" * 64,
        dispatch_key="trt_engine",
        features={
            "supports_fp16_tensorcore": 1.0,
            "supports_int8_tensorcore": 1.0,
            "int8_precision_propagation_ratio": 0.95,
            "qdq_fold_ratio": 0.9,
            "reformat_rate": 0.03,
        },
    ),
]


def _probe_rows(decision: dict[str, Any]) -> list[dict[str, Any]]:
    ratios = {
        "h800-tvm-auto-v3": float(decision["tvm_top2_region"]["int8_median_ms"])
        / float(decision["tvm_top2_region"]["fp16_median_ms"]),
        "h800-trt-v3": float(decision["trt_32x64x128_backbone"]["int8_p50_ms"])
        / float(decision["trt_32x64x128_backbone"]["fp16_p50_ms"]),
    }
    rows = []
    for width, scale in [([16, 32, 64], 1.0), ([32, 64, 128], 2.0), ([48, 96, 192], 3.0)]:
        for profile_id, int8_ratio in ratios.items():
            for q_mode, ratio in (("fp16", 1.0), ("int8", int8_ratio)):
                rows.append(
                    {
                        "row_id": f"probe|{profile_id}|{'x'.join(map(str, width))}|{q_mode}",
                        "width": width,
                        "q_mode": q_mode,
                        "capability_profile_id": profile_id,
                        "graph_features": {"normalized_scale": scale},
                        "build_status": "success",
                        "numerical_status": "pass",
                        "latency_ms": scale * ratio,
                        "energy_j": scale * ratio,
                        "ap70": 0.6,
                    }
                )
    return rows


def run_smoke(*, profiles_path: Path, decision_path: Path) -> dict[str, Any]:
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    rows = _probe_rows(decision)
    bundle = fit_model_bundle(rows, profiles, ridge=1e-6)
    candidates = [row for row in rows if row["width"] == [32, 64, 128]]
    conditioned = predict_rows(bundle, candidates, profiles)
    blind = predict_rows(backend_blind_bundle(bundle), candidates, profiles)
    return {
        "schema_version": "stage2_canonical_search_smoke_v3",
        "evidence_kind": "historical_label_mechanism_test",
        "valid_as_cold_start_framework_evidence": False,
        "trusted_for_final_frontier": False,
        "note": "Historical performance labels are used only to test code mechanics; this is not a leakage-free framework validation.",
        "training_rows": len(rows),
        "conditioned_decisions": best_q_by_profile(conditioned, objective="latency_ms"),
        "backend_blind_decisions": best_q_by_profile(blind, objective="latency_ms"),
        "unconditional_int8_regret": unconditional_int8_regret(
            conditioned, objective="latency_ms"
        ),
        "feature_schema_sha256": bundle["feature_schema"]["feature_schema_sha256"],
        "training_data_sha256": bundle["training_data_sha256"],
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-json", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--decision-json", type=Path, default=DEFAULT_DECISION)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_smoke(profiles_path=args.profiles_json, decision_path=args.decision_json)
    write_report(report, args.out_json)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
