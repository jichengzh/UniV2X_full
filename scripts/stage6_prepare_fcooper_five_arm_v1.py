#!/usr/bin/env python3
"""Freeze F-Cooper backend-neutral controls for the five-arm TRT table."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage5.single_target_search_v2 import (  # noqa: E402
    SearchTask,
    build_measurement_request,
)


GRAPH_FEATURES = (
    "parameter_elements",
    "conv_flops",
    "conv_count",
    "group_conv_count",
    "stride2_conv_count",
    "arithmetic_intensity_proxy",
)


def _rows(path: Path, field: str = "rows") -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    rows = payload.get(field)
    if not isinstance(rows, list):
        raise ValueError(f"expected rows in {path}")
    return [dict(row) for row in rows]


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def genome_key(row: Mapping[str, Any]) -> tuple[int, int, int, int, int, str]:
    width = tuple(int(value) for value in row.get("width") or [])
    if len(width) != 5:
        raise ValueError("F-Cooper candidate must expose five scanner widths")
    return (*width, str(row["q_mode"]))


def _normalize(values: Sequence[float], value: float) -> float:
    low, high = min(values), max(values)
    return 0.0 if high == low else (value - low) / (high - low)


def _q_bits(q_mode: str) -> int:
    return {"fp16": 16, "int8": 8}[q_mode]


def rank_backend_neutral_candidates(
    candidates: Sequence[Mapping[str, Any]],
    ap_by_genome: Mapping[tuple[int, int, int, int, int, str], float],
    graph_costs: Mapping[tuple[int, int, int, int, int], Mapping[str, float]],
) -> list[dict[str, Any]]:
    rows = []
    for source in candidates:
        row = dict(source)
        key = genome_key(row)
        width_key = key[:5]
        costs = graph_costs[width_key]
        bits = _q_bits(key[-1])
        rows.append(
            {
                "genome": [*width_key, key[-1]],
                "row_id": row["row_id"],
                "ap_surrogate": float(ap_by_genome[key]),
                "parameter_elements": float(costs["parameter_elements"]),
                "conv_flops": float(costs["conv_flops"]),
                "q_bits": bits,
                "parameter_bits": float(costs["parameter_elements"]) * bits,
                "bitops": float(costs["conv_flops"]) * bits,
                "source_evidence_sha256": row["source_evidence_sha256"],
            }
        )
    columns = {
        name: [float(row[name]) for row in rows]
        for name in ("ap_surrogate", "parameter_bits", "bitops")
    }
    ranked = []
    for row in rows:
        score = (
            _normalize(columns["ap_surrogate"], row["ap_surrogate"])
            - 0.5 * _normalize(columns["parameter_bits"], row["parameter_bits"])
            - 0.5 * _normalize(columns["bitops"], row["bitops"])
        )
        ranked.append({**row, "hardware_blind_acquisition_score": score})
    return sorted(
        ranked,
        key=lambda row: (-row["hardware_blind_acquisition_score"], row["row_id"]),
    )


def _width_features(width: Sequence[int]) -> list[float]:
    values = [float(value) for value in width]
    return [
        1.0,
        *values,
        *[value * value for value in values],
        *[
            values[left] * values[right]
            for left in range(len(values))
            for right in range(left + 1, len(values))
        ],
    ]


def fit_graph_costs(
    observed_graphs: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int, int, int, int], dict[str, float]]:
    usable = [
        row
        for row in observed_graphs
        if len(row.get("width") or []) == 5
        and all(math.isfinite(float(row.get(name, math.nan))) for name in ("parameter_elements", "conv_flops"))
    ]
    if len({tuple(row["width"]) for row in usable}) < 8:
        raise ValueError("at least eight actual F-Cooper graphs are required")
    x_train = np.asarray([_width_features(row["width"]) for row in usable])
    models = {}
    for offset, target in enumerate(("parameter_elements", "conv_flops")):
        model = ExtraTreesRegressor(
            n_estimators=512,
            min_samples_leaf=1,
            max_features=1.0,
            random_state=20260723 + offset,
            n_jobs=1,
        )
        model.fit(x_train, np.log1p([float(row[target]) for row in usable]))
        models[target] = model
    widths = sorted({tuple(int(value) for value in row["width"]) for row in candidates})
    x_predict = np.asarray([_width_features(width) for width in widths])
    predictions = {
        target: np.expm1(model.predict(x_predict))
        for target, model in models.items()
    }
    return {
        width: {
            target: float(max(0.0, predictions[target][index]))
            for target in predictions
        }
        for index, width in enumerate(widths)
    }


def _ap_feature(
    row: Mapping[str, Any], graph: Mapping[str, Any]
) -> list[float]:
    width = [float(value) for value in row["width"]]
    width = [*width, *([0.0] * (5 - len(width)))]
    return [
        *width[:5],
        float(row["q_mode"] == "int8"),
        float(row["model"] == "codriving"),
        float(row["model"] == "fcooper"),
        *[float(graph.get(name) or 0.0) for name in GRAPH_FEATURES],
    ]


def fit_neutral_ap(
    training_rows: Sequence[Mapping[str, Any]],
    training_graphs: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    fcooper_ap70_ref: float,
    fcooper_original_graph: Mapping[str, Any],
) -> dict[tuple[int, int, int, int, int, str], float]:
    graph_by_group = {str(row["group_id"]): row for row in training_graphs}
    x_train, y_train = [], []
    for row in training_rows:
        if row.get("terminal_status") != "measured_success_gold":
            continue
        graph = graph_by_group.get(str(row["group_id"]))
        if graph is None:
            continue
        x_train.append(_ap_feature(row, graph))
        y_train.append(float(row["ap70"]))
    anchor = {
        "model": "fcooper",
        "width": [64, 128, 256, 128, 256],
        "q_mode": "fp32",
    }
    x_train.append(_ap_feature(anchor, fcooper_original_graph))
    y_train.append(float(fcooper_ap70_ref))
    model = ExtraTreesRegressor(
        n_estimators=512,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=20260723,
        n_jobs=1,
    )
    model.fit(np.asarray(x_train), np.asarray(y_train))
    predictions = model.predict(
        np.asarray(
            [
                _ap_feature(row, row.get("graph_features") or {})
                for row in candidates
            ]
        )
    )
    return {
        genome_key(row): float(value)
        for row, value in zip(candidates, predictions)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--observed-graphs-json", type=Path, required=True)
    parser.add_argument("--probe-graphs-json", type=Path, required=True)
    parser.add_argument("--coldstart-rows-json", type=Path, required=True)
    parser.add_argument("--coldstart-graphs-json", type=Path, required=True)
    parser.add_argument("--profiles-json", type=Path, required=True)
    parser.add_argument("--frozen-contract-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = _rows(args.candidate_json)
    observed = _rows(args.observed_graphs_json)
    probe_graphs = _rows(args.probe_graphs_json, field="graph_features")
    graph_costs = fit_graph_costs([*probe_graphs, *observed], candidates)
    contract = json.loads(args.frozen_contract_json.read_text())
    original_graph = next(
        row
        for row in probe_graphs
        if list(row["width"]) == [64, 128, 256, 128, 256]
    )
    ap_by_genome = fit_neutral_ap(
        _rows(args.coldstart_rows_json),
        _rows(args.coldstart_graphs_json, field="graph_features"),
        candidates,
        fcooper_ap70_ref=float(contract["ap70_ref"]),
        fcooper_original_graph=original_graph,
    )
    ranked = rank_backend_neutral_candidates(candidates, ap_by_genome, graph_costs)
    by_key = {genome_key(row): row for row in candidates}
    compression = [row["genome"] for row in ranked[:16]]
    screened = [row["genome"] for row in ranked[:12]]
    locked = [row["genome"] for row in ranked[:4]]
    plan = {
        "schema_version": "stage6_fcooper_five_arm_plan_v1",
        "target_model": "fcooper",
        "hardware_id": "h800",
        "backend": "trt",
        "hardware_blind_backend_labels_used": False,
        "hardware_blind_cost_policy": "actual_graph_conditioned_parameter_bits_and_bitops_surrogate",
        "candidate_pool_size": len(candidates),
        "ranked_candidates": ranked,
        "arms": {
            "original_default": {
                "fixed_genome": [64, 128, 256, 128, 256, "fp32"]
            },
            "compression_only": {
                "selected_genomes": compression,
                "outer_budget": 16,
                "builder_optimization_level": 0,
            },
            "schedule_only": {
                "fixed_genome": [64, 128, 256, 128, 256, "fp32"],
                "builder_optimization_level": 5,
            },
            "compress_then_tune": {
                "screened_genomes": screened,
                "locked_genomes": locked,
                "outer_budget": {"screen": 12, "locked": 4},
                "builder_optimization_level": 5,
            },
            "gear": {"actual_feedback_budget": 16, "builder_optimization_level": 5},
        },
    }
    _write(args.output_dir / "stage6_fcooper_five_arm_plan.json", plan)
    profiles_payload = json.loads(args.profiles_json.read_text())
    profiles = (
        profiles_payload.get("capability_profiles")
        if isinstance(profiles_payload, dict)
        else profiles_payload
    )
    if not isinstance(profiles, list):
        raise ValueError("capability profiles must be a list")
    profile = next(row for row in profiles if row["dispatch_key"] == "trt_engine")
    task = SearchTask("S5-FCO-TRT", "fcooper", "h800", profile)
    for arm, genomes in (
        ("compression_only", compression),
        ("compress_then_tune", locked),
    ):
        selected = [by_key[tuple(genome)] for genome in genomes]
        _write(
            args.output_dir / arm / "candidate_plan.json",
            {
                "schema_version": "stage6_fcooper_arm_candidate_plan_v1",
                "arm_id": arm,
                "rows": selected,
            },
        )
        for offset in range(0, len(selected), 4):
            request = build_measurement_request(
                task=task,
                selected_rows=selected[offset : offset + 4],
                round_index=offset // 4,
            )
            _write(
                args.output_dir
                / arm
                / f"batch_{offset // 4:02d}"
                / "measurement_request.json",
                request,
            )
    print(
        json.dumps(
            {
                "candidate_pool": len(candidates),
                "compression_only": len(compression),
                "compress_then_tune": len(locked),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
