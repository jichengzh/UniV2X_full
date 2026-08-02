#!/usr/bin/env python3
"""Evaluate leakage-free C0 static versus C1 structural capability context."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from framework.stage2.canonical_search_v3 import build_capability_profile
from framework.stage2.cost_model_bundle_v3 import fit_model_bundle, predict_rows


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _pooled_features(records: list[dict[str, Any]]) -> dict[str, float]:
    int8 = [row for row in records if row["q_mode"] == "int8"]
    if len(records) != 12 or len(int8) != 6 or not all(row["build_success"] for row in records):
        raise ValueError("S1 evaluation requires six successful INT8 and twelve successful total probes")

    def ratio(numerator: str, denominator: str) -> float:
        top = sum(float(row[numerator]) for row in int8)
        bottom = sum(float(row[denominator]) for row in int8)
        return top / bottom

    return {
        "compiler_structural_probe_observed": 1.0,
        "build_success_rate": sum(row["build_success"] for row in records) / len(records),
        "int8_precision_propagation_ratio": ratio(
            "int8_propagated_ops", "precision_eligible_ops"
        ),
        "qdq_fold_ratio": ratio("qdq_folded_pairs", "qdq_pairs"),
        # Reformat/fusion extraction remains inspector-specific diagnostic evidence.
        # Only the directly verified Conv dtype and Q/DQ boundary ratios enter C1.
    }


def _profiles(features_by_profile: dict[str, dict[str, float]], *, c1: bool) -> list[dict[str, Any]]:
    common = {
        "supports_fp16_tensorcore": 1.0,
        "supports_int8_tensorcore": 1.0,
        "int8_channel_alignment": 16.0,
        "fp16_channel_alignment": 8.0,
    }
    dispatch = {"tvm-profile": "tvm_auto", "trt-profile": "trt_engine"}
    return [
        build_capability_profile(
            capability_profile_id=profile_id,
            hardware_target="h800",
            compiler_fingerprint=_digest(profile_id),
            dispatch_key=dispatch[profile_id],
            features={**common, **(features if c1 else {})},
        )
        for profile_id, features in features_by_profile.items()
    ]


def _gold_rows(path: Path) -> list[dict[str, Any]]:
    source = json.loads(path.read_text(encoding="utf-8"))["rows"]
    profile_map = {
        "h800_tvm_routeb_20260708": "tvm-profile",
        "h800_trt_20260708": "trt-profile",
    }
    rows = []
    for item in source:
        if item.get("compiler_profile_id") not in profile_map:
            continue
        if item.get("q_mode") not in {"fp16", "int8"}:
            continue
        if item.get("mixed_policy_id") not in {"none", "all_eligible_conv"}:
            continue
        if not all(isinstance(item.get(name), (int, float)) for name in ("latency_ms", "energy_j")):
            continue
        width = [int(value) for value in str(item["width"]).split("x")]
        rows.append(
            {
                "row_id": f"{item['compiler_profile_id']}|{item['width']}|{item['q_mode']}",
                "width": width,
                "q_mode": item["q_mode"],
                "capability_profile_id": profile_map[item["compiler_profile_id"]],
                "graph_features": {},
                "build_status": "success",
                "numerical_status": "pass",
                "latency_ms": float(item["latency_ms"]),
                "energy_j": float(item["energy_j"]),
                "ap70": 0.0,
            }
        )
    return rows


def _evaluate(
    rows: list[dict[str, Any]], profiles: list[dict[str, Any]], *, split_seed: int
) -> dict[str, Any]:
    widths = sorted({tuple(row["width"]) for row in rows})
    ranked = sorted(
        widths, key=lambda width: _digest(f"{split_seed}:" + "x".join(map(str, width)))
    )
    holdout = set(ranked[:4])
    train = [row for row in rows if tuple(row["width"]) not in holdout]
    test = [row for row in rows if tuple(row["width"]) in holdout]
    bundle = fit_model_bundle(train, profiles, ridge=1e-4)
    predicted = predict_rows(bundle, test, profiles)
    metrics = {}
    for objective in ("latency_ms", "energy_j"):
        errors = [
            abs(float(row["predictions"][objective]) - float(row[objective])) / float(row[objective])
            for row in predicted
        ]
        metrics[f"{objective}_mape"] = sum(errors) / len(errors)
    correct = 0
    total = 0
    grouped: dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]] = {}
    for row in predicted:
        key = (row["capability_profile_id"], tuple(row["width"]))
        grouped.setdefault(key, []).append(row)
    for pair in grouped.values():
        actual = min(pair, key=lambda row: float(row["latency_ms"]))["q_mode"]
        predicted_q = min(pair, key=lambda row: float(row["predictions"]["latency_ms"]))["q_mode"]
        correct += actual == predicted_q
        total += 1
    metrics["precision_choice_accuracy"] = correct / total
    return {
        "train_rows": len(train),
        "holdout_rows": len(test),
        "holdout_widths": [list(width) for width in sorted(holdout)],
        **metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tvm-records", type=Path, required=True)
    parser.add_argument("--trt-records", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tvm_run = json.loads(args.tvm_records.read_text(encoding="utf-8"))
    trt_run = json.loads(args.trt_records.read_text(encoding="utf-8"))
    features = {
        "tvm-profile": _pooled_features(tvm_run["records"]),
        "trt-profile": _pooled_features(trt_run["records"]),
    }
    rows = _gold_rows(args.gold)
    c0_profiles = _profiles(features, c1=False)
    c1_profiles = _profiles(features, c1=True)
    c0 = _evaluate(rows, c0_profiles, split_seed=42)
    c1 = _evaluate(rows, c1_profiles, split_seed=42)
    robustness = []
    for seed in range(20):
        c0_seed = _evaluate(rows, c0_profiles, split_seed=seed)
        c1_seed = _evaluate(rows, c1_profiles, split_seed=seed)
        robustness.append(
            {
                "seed": seed,
                "c0_precision_choice_accuracy": c0_seed["precision_choice_accuracy"],
                "c1_precision_choice_accuracy": c1_seed["precision_choice_accuracy"],
                "c0_latency_mape": c0_seed["latency_ms_mape"],
                "c1_latency_mape": c1_seed["latency_ms_mape"],
            }
        )
    report = {
        "schema_version": "stage2_s1_c0_c1_eval_v3",
        "evidence_scope": "codriving_old_v2_latency_energy_replay_only",
        "ap_excluded_reason": "TVM INT8 AP rows in this table are known collapsed/invalid",
        "gold_rows": len(rows),
        "s1_features": features,
        "probe_cost_seconds": {
            "tvm": float(tvm_run["wall_seconds"]),
            "trt": float(trt_run["wall_seconds"]),
            "total": float(tvm_run["wall_seconds"]) + float(trt_run["wall_seconds"]),
        },
        "c0": c0,
        "c1": c1,
        "robustness_20_grouped_splits": {
            "c0_mean_precision_choice_accuracy": sum(
                row["c0_precision_choice_accuracy"] for row in robustness
            )
            / len(robustness),
            "c1_mean_precision_choice_accuracy": sum(
                row["c1_precision_choice_accuracy"] for row in robustness
            )
            / len(robustness),
            "c1_better_choice_split_count": sum(
                row["c1_precision_choice_accuracy"] > row["c0_precision_choice_accuracy"]
                for row in robustness
            ),
            "c1_lower_latency_mape_split_count": sum(
                row["c1_latency_mape"] < row["c0_latency_mape"] for row in robustness
            ),
            "splits": robustness,
        },
        "delta": {
            "precision_choice_accuracy_points": 100.0
            * (c1["precision_choice_accuracy"] - c0["precision_choice_accuracy"]),
            "latency_mape_relative_reduction": 1.0
            - c1["latency_ms_mape"] / c0["latency_ms_mape"],
            "energy_mape_relative_reduction": 1.0
            - c1["energy_j_mape"] / c0["energy_j_mape"],
        },
        "decision": (
            "C0_insufficient_C1_useful_for_seen_profiles"
            if c1["precision_choice_accuracy"] > c0["precision_choice_accuracy"]
            else "C0_not_disproved"
        ),
        "generalization_limit": (
            "Width-grouped replay contains both compiler profiles in train and holdout; "
            "it does not prove transfer to an unseen compiler profile."
        ),
        "s2_status": "deferred_pending_new_gold_and_unseen_profile_validation",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
