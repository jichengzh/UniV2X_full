#!/usr/bin/env python3
"""Phase1 QxS search entry.

This entry keeps the automatic-search narrative explicit: the genome is
``(w0,w1,w2,search_arm_id)`` and historical hand-rewrite data can be used as a
warm-start prior, but cannot enter the final Pareto gold set.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO = Path("/home/jichengzhi/V2X")
DEFAULT_MANIFEST = REPO / "results/phase1_qxs_search_space_manifest_20260707.json"
DEFAULT_TRAINING_TABLE = REPO / "results/original60_training_table_backend_relabel_20260707.json"
DEFAULT_KEYPOINTS = REPO / "results/int8_arm_routeb_keypoints_20260707.json"
DEFAULT_OUT = REPO / "results/phase1_qxs_smbo_entry_20260707.json"

WIDTH_GRIDS = (
    [16, 24, 32, 40, 48, 56, 64],
    [32, 48, 64, 80, 96, 112, 128],
    [64, 96, 128, 160, 192, 224, 256],
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_width(value: Any) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)):
        parts = [int(item) for item in value]
    elif isinstance(value, str):
        text = value.replace("x", ",")
        parts = [int(item.strip()) for item in text.split(",") if item.strip()]
    else:
        raise TypeError(f"unsupported width value: {value!r}")
    if len(parts) != 3:
        raise ValueError(f"width must have 3 parts, got {value!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def width_str(width: Any) -> str:
    w0, w1, w2 = parse_width(width)
    return f"{w0}x{w1}x{w2}"


def table_rows(table: Any) -> list[dict[str, Any]]:
    if isinstance(table, list):
        return list(table)
    return list(table.get("rows", []))


def manifest_arms(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return list(manifest.get("arms", []))


def _ordered_unique(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _arm_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(arm["search_arm_id"]): arm for arm in manifest_arms(manifest)}


def _row_q_axis(row: dict[str, Any], arms: dict[str, dict[str, Any]]) -> str:
    if row.get("q_axis"):
        return str(row["q_axis"])
    arm = arms.get(str(row.get("search_arm_id")))
    if arm and arm.get("q_axis"):
        return str(arm["q_axis"])
    precision = str(row.get("precision_original") or row.get("precision") or "unknown")
    return "int8" if precision in {"int8", "int8_tc"} else precision


def _row_backend(row: dict[str, Any], arms: dict[str, dict[str, Any]]) -> str:
    if row.get("backend"):
        return str(row["backend"])
    arm = arms.get(str(row.get("search_arm_id")))
    if arm and arm.get("backend"):
        return str(arm["backend"])
    return "unknown_backend"


def _source_confidence(row: dict[str, Any]) -> float:
    value = row.get("confidence_weight", row.get("source_confidence", 1.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def build_feature_spec(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    arms = _arm_by_id(manifest)
    q_axes = _ordered_unique(
        [arm.get("q_axis") for arm in manifest_arms(manifest)]
        + [_row_q_axis(row, arms) for row in rows]
    )
    backends = _ordered_unique(
        [arm.get("backend") for arm in manifest_arms(manifest)]
        + [_row_backend(row, arms) for row in rows]
    )
    feature_order = (
        ["w0", "w1", "w2"]
        + [f"q_axis={item}" for item in q_axes]
        + [f"backend={item}" for item in backends]
        + ["source_confidence"]
    )
    return {
        "genome_schema": ["w0", "w1", "w2", "search_arm_id"],
        "feature_order": feature_order,
        "q_axis_categories": q_axes,
        "backend_categories": backends,
    }


def encode_training_row(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    w0, w1, w2 = parse_width(row.get("width", [row.get("w0"), row.get("w1"), row.get("w2")]))
    arm_id = str(row.get("search_arm_id", "unknown_arm"))
    q_axis = str(row.get("q_axis") or row.get("precision_original") or row.get("precision") or "unknown")
    if q_axis == "int8_tc":
        q_axis = "int8"
    backend = str(row.get("backend", "unknown_backend"))
    values: list[float] = [float(w0), float(w1), float(w2)]
    values.extend(1.0 if item == q_axis else 0.0 for item in spec["q_axis_categories"])
    values.extend(1.0 if item == backend else 0.0 for item in spec["backend_categories"])
    values.append(_source_confidence(row))
    return {
        "label": row.get("label"),
        "width": f"{w0}x{w1}x{w2}",
        "genome": [w0, w1, w2, arm_id],
        "features": values,
        "target": {
            "latency_ms": row.get("latency_ms"),
            "energy_j": row.get("energy_j"),
            "ap70": row.get("ap70"),
        },
        "source": {
            "search_arm_id": arm_id,
            "q_axis": q_axis,
            "backend": backend,
            "trusted_for_final_frontier": row.get("trusted_for_final_frontier"),
            "source_class": row.get("source_class"),
            "evidence_kind": row.get("evidence_kind"),
        },
    }


def _frontier_flag_ok(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.lower() in {"true", "conditional"}
    return False


def _can_arm_enter_final(arm: dict[str, Any]) -> bool:
    return _frontier_flag_ok(arm.get("can_enter_final_frontier"))


def _is_historical_prior_arm(arm_id: str) -> bool:
    return "historical_hand_rewrite_prior" in str(arm_id) or "hand_rewrite_prior" in str(arm_id)


def allowed_final_arm_ids(manifest: dict[str, Any]) -> set[str]:
    return {
        str(arm["search_arm_id"])
        for arm in manifest_arms(manifest)
        if _can_arm_enter_final(arm) and not _is_historical_prior_arm(str(arm["search_arm_id"]))
    }


def filter_final_frontier_rows(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    pipeline_arm_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    allowed = set(pipeline_arm_ids or allowed_final_arm_ids(manifest))
    final_rows: list[dict[str, Any]] = []
    for row in rows:
        arm_id = str(row.get("search_arm_id", ""))
        if _is_historical_prior_arm(arm_id):
            continue
        if arm_id not in allowed:
            continue
        if not _frontier_flag_ok(row.get("trusted_for_final_frontier")):
            continue
        final_rows.append(dict(row))
    return final_rows


def warm_start_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("latency_ms") is None:
            continue
        if row.get("can_use_as_cold_start_prior", True) is False:
            continue
        out.append(dict(row))
    return out


def _axis_neighbors(width: tuple[int, int, int], width_grids: tuple[list[int], list[int], list[int]]) -> list[tuple[int, int, int]]:
    neighbors: list[tuple[int, int, int]] = []
    for axis, grid in enumerate(width_grids):
        value = width[axis]
        if value not in grid:
            continue
        idx = grid.index(value)
        for next_idx in (idx - 1, idx + 1):
            if 0 <= next_idx < len(grid):
                cand = list(width)
                cand[axis] = int(grid[next_idx])
                neighbors.append((cand[0], cand[1], cand[2]))
    return neighbors


def build_disagreement_remeasure_plan(
    keypoint_rows: list[dict[str, Any]],
    *,
    width_grids: tuple[list[int], list[int], list[int]] = WIDTH_GRIDS,
    threshold: float = 1.5,
) -> dict[str, Any]:
    centers: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in keypoint_rows:
        ratio = row.get("routeb_div_old_int8tc_prior")
        try:
            ratio_value = float(ratio)
        except (TypeError, ValueError):
            continue
        if ratio_value <= float(threshold):
            continue
        center = parse_width(row["width"])
        center_text = width_str(center)
        centers.append(
            {
                "width": center_text,
                "routeb_div_old_int8tc_prior": ratio_value,
                "trigger": "routeb_auto_vs_historical_prior_ratio",
            }
        )
        for width, relation in [(center, "center")] + [(item, "axis_neighbor") for item in _axis_neighbors(center, width_grids)]:
            text = width_str(width)
            if text in seen:
                continue
            seen.add(text)
            queue.append(
                {
                    "width": text,
                    "center_width": center_text,
                    "relation": relation,
                    "reason": f"RouteB_auto/historical_prior>{float(threshold):.3g}",
                }
            )
    return {"threshold": float(threshold), "centers": centers, "remeasure_queue": queue}


def build_entry(
    manifest: dict[str, Any],
    training_table: dict[str, Any] | list[dict[str, Any]],
    keypoints: dict[str, Any] | list[dict[str, Any]] | None = None,
    *,
    disagreement_threshold: float = 1.5,
) -> dict[str, Any]:
    rows = table_rows(training_table)
    keypoint_rows = table_rows(keypoints or [])
    feature_spec = build_feature_spec(manifest, rows)
    encoded_rows = [encode_training_row(row, feature_spec) for row in rows]
    final_rows = filter_final_frontier_rows(rows, manifest)
    warm_rows = warm_start_rows(rows)
    disagreement = build_disagreement_remeasure_plan(
        keypoint_rows,
        width_grids=WIDTH_GRIDS,
        threshold=disagreement_threshold,
    )
    historical_rows = [row for row in rows if _is_historical_prior_arm(str(row.get("search_arm_id", "")))]
    return {
        "schema_version": "phase1_qxs_smbo_entry_v1",
        "created_at_utc": utc_now(),
        "inputs": {
            "manifest_schema": manifest.get("schema_version"),
            "training_table_schema": training_table.get("schema_version") if isinstance(training_table, dict) else None,
        },
        "genome_schema": feature_spec["genome_schema"],
        "feature_order": feature_spec["feature_order"],
        "q_axis_categories": feature_spec["q_axis_categories"],
        "backend_categories": feature_spec["backend_categories"],
        "policy": {
            "final_frontier_rule": "trusted_for_final_frontier true/conditional AND active manifest arm; historical hand-rewrite prior excluded",
            "historical_hand_rewrite_prior": "warm_start_only_not_final_gold",
            "disagreement_trigger": f"RouteB_auto/historical_prior>{float(disagreement_threshold):.3g}",
        },
        "summary": {
            "n_manifest_arms": len(manifest_arms(manifest)),
            "n_training_rows": len(rows),
            "n_encoded_training_rows": len(encoded_rows),
            "n_warm_start_rows": len(warm_rows),
            "n_final_frontier_gold_rows": len(final_rows),
            "n_historical_prior_rows": len(historical_rows),
            "n_disagreement_centers": len(disagreement["centers"]),
            "n_remeasure_queue": len(disagreement["remeasure_queue"]),
        },
        "allowed_final_arm_ids": sorted(allowed_final_arm_ids(manifest)),
        "training_feature_rows": encoded_rows,
        "warm_start_rows": warm_rows,
        "final_frontier_gold_rows": final_rows,
        "disagreement": disagreement,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--training-table", type=Path, default=DEFAULT_TRAINING_TABLE)
    parser.add_argument("--keypoints", type=Path, default=DEFAULT_KEYPOINTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--disagreement-threshold", type=float, default=1.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_json(args.manifest)
    training_table = load_json(args.training_table)
    keypoints = load_json(args.keypoints) if args.keypoints.is_file() else []
    result = build_entry(
        manifest,
        training_table,
        keypoints,
        disagreement_threshold=args.disagreement_threshold,
    )
    write_json(args.out, result)
    print(json.dumps({"out": str(args.out), "summary": result["summary"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
