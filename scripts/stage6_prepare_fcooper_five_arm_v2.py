#!/usr/bin/env python3
"""Prepare the formal F-Cooper five-arm Stage6 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage5.single_target_search_v2 import (  # noqa: E402
    SearchTask,
    build_task_candidate_manifest,
    freeze_initial_coldstart,
    verify_frozen_coldstart_artifacts,
)
from scripts.stage5_advance_fcooper_round_v1 import (  # noqa: E402
    validate_fcooper_registry,
)
from scripts.stage6_prepare_fcooper_five_arm_v1 import (  # noqa: E402
    fit_graph_costs,
    fit_neutral_ap,
    genome_key,
    rank_backend_neutral_candidates,
)


TASK_ID = "S5-FCO-TRT-V2"
PILOT_FRAGMENT = "fcooper_workpackage_a_20260723"
BASE_WIDTH = (64, 128, 256, 128, 256)
SUCCESS = "measured_success_gold"
TERMINAL_FAILURES = {"feasibility_failure", "numerical_feasibility_failure"}
CONTROL_TASK_IDS = {
    ("schedule_only", "measure"): "S6-FCO-TRT-SCHEDULE-ONLY-MEASURE-V2",
    ("compression_only", "measure"): "S6-FCO-TRT-COMPRESSION-ONLY-MEASURE-V2",
    ("compress_then_tune", "screen"): (
        "S6-FCO-TRT-COMPRESS-THEN-TUNE-SCREEN-V2"
    ),
    ("compress_then_tune", "tuned_remeasurement"): (
        "S6-FCO-TRT-COMPRESS-THEN-TUNE-TUNED-V2"
    ),
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(payload: Any, field: str = "rows") -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, Mapping) and isinstance(payload.get(field), list):
        return [dict(row) for row in payload[field]]
    raise ValueError(f"expected list or object containing {field}")


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contains_pilot_reference(value: Any) -> bool:
    if isinstance(value, str):
        return PILOT_FRAGMENT in value
    if isinstance(value, Mapping):
        return any(
            _contains_pilot_reference(key) or _contains_pilot_reference(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_pilot_reference(item) for item in value)
    return False


def reject_pilot_reference(value: Any, *, context: str) -> None:
    if _contains_pilot_reference(value):
        raise ValueError(f"{context} contains forbidden pilot path fragment")


def load_frozen_gold176(
    rows_path: Path, graph_features_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    try:
        audit = verify_frozen_coldstart_artifacts(rows_path, graph_features_path)
        rows = freeze_initial_coldstart(_rows(_read_json(rows_path)))
        graphs = _rows(_read_json(graph_features_path), "graph_features")
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Stage6 requires byte-frozen standalone Gold176: {error}") from error
    reject_pilot_reference(rows, context="Gold176 rows")
    reject_pilot_reference(graphs, context="Gold176 graph features")
    return rows, graphs, audit


def _task(profile: Mapping[str, Any]) -> SearchTask:
    return SearchTask(TASK_ID, "fcooper", "h800", profile)


def candidates_from_scanner_registry(
    registry: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expand exactly the structures present in the scanner source registry."""
    reject_pilot_reference(registry, context="scanner registry")
    validate_fcooper_registry(registry)
    manifest = build_task_candidate_manifest(
        registry,
        task=_task(profile),
        measured_row_ids=set(),
    )
    rows = [dict(row) for row in manifest["rows"]]
    if not rows:
        raise ValueError("scanner registry produced an empty F-Cooper pool")
    if any(row.get("task_id") != TASK_ID for row in rows):
        raise ValueError("scanner candidate task identity drift")
    return rows


def build_five_arm_protocol(
    ranked_candidates: Sequence[Mapping[str, Any]],
    *,
    candidate_pool_size: int,
) -> dict[str, Any]:
    if len(ranked_candidates) < 16:
        raise ValueError("formal controls require at least 16 scanner candidates")
    ranked = [dict(row) for row in ranked_candidates]
    row_ids = [str(row.get("row_id") or "") for row in ranked]
    if any(not row_id for row_id in row_ids) or len(set(row_ids)) != len(row_ids):
        raise ValueError("ranked scanner candidates need unique row_id values")
    compression = ranked[:16]
    screen = ranked[:12]
    return {
        "schema_version": "stage6_fcooper_five_arm_plan_v2",
        "task_id": TASK_ID,
        "target_model": "fcooper",
        "hardware_id": "h800",
        "backend": "trt",
        "candidate_source": "fresh_scanner_source_registry",
        "candidate_pool_size": int(candidate_pool_size),
        "hardware_blind_backend_labels_used": False,
        "hardware_blind_cost_policy": (
            "formal_actual_graph_conditioned_parameter_bits_and_bitops_surrogate"
        ),
        "ranked_candidates": ranked,
        "arms": {
            "original_default": {
                "fixed_width": list(BASE_WIDTH),
                "fixed_q_mode": "fp32",
                "builder_optimization_level": 0,
                "outer_budget": 1,
            },
            "compression_only": {
                "control_task_id": CONTROL_TASK_IDS[
                    ("compression_only", "measure")
                ],
                "selection_policy": "hardware_blind_algorithmic_rank_top16",
                "selected_row_ids": [row["row_id"] for row in compression],
                "selected_genomes": [row["genome"] for row in compression],
                "outer_budget": 16,
                "builder_optimization_level": 0,
                "pruned_success_requires_recovery_training_evidence": True,
            },
            "schedule_only": {
                "control_task_id": CONTROL_TASK_IDS[("schedule_only", "measure")],
                "fixed_width": list(BASE_WIDTH),
                "fixed_q_mode": "fp32",
                "builder_optimization_level": 5,
                "outer_budget": 1,
            },
            "compress_then_tune": {
                "screen_control_task_id": CONTROL_TASK_IDS[
                    ("compress_then_tune", "screen")
                ],
                "tuned_control_task_id": CONTROL_TASK_IDS[
                    ("compress_then_tune", "tuned_remeasurement")
                ],
                "protocol": "fixed_sequential_12_screen_plus_4_tuned_remeasurement",
                "screen_selection_policy": "hardware_blind_algorithmic_rank_top12",
                "tuned_selection_policy": (
                    "post_screen_ap_feasible_latency_then_energy_top4"
                ),
                "screen_row_ids": [row["row_id"] for row in screen],
                "screen_genomes": [row["genome"] for row in screen],
                "tuned_row_ids": [],
                "screen_budget": 12,
                "tuned_remeasurement_budget": 4,
                "screen_builder_optimization_level": 0,
                "tuned_builder_optimization_level": 5,
                "phase_status": "awaiting_12_screen_observations",
            },
            "gear": {
                "candidate_source": "formal_t16_online_feedback_only",
                "task_id": TASK_ID,
                "actual_feedback_budget": 16,
                "builder_optimization_level": 5,
                "probe_rows_allowed": False,
            },
        },
    }


def _finite_metric(row: Mapping[str, Any], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"screen feedback has non-finite {name}")
    return value


def _has_sha(row: Mapping[str, Any], name: str) -> bool:
    value = row.get(name)
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def select_tuned_remeasurements(
    screen_feedback: Sequence[Mapping[str, Any]],
    *,
    screen_row_ids: Sequence[str],
    ap70_ref: float,
    max_ap_drop: float,
) -> list[dict[str, Any]]:
    """Choose the tuned four only after the fixed 12-row screen is observed."""
    expected_task_id = CONTROL_TASK_IDS[("compress_then_tune", "screen")]
    expected_ids = [str(row_id) for row_id in screen_row_ids]
    rows = [dict(row) for row in screen_feedback]
    observed_ids = [str(row.get("row_id") or "") for row in rows]
    if (
        len(expected_ids) != 12
        or len(set(expected_ids)) != 12
        or len(rows) != 12
        or len(set(observed_ids)) != 12
        or set(observed_ids) != set(expected_ids)
    ):
        raise ValueError("compress-then-tune requires exactly 12 fixed screen observations")
    for row in rows:
        if row.get("task_id") != expected_task_id:
            raise ValueError("screen feedback task identity drift")
        status = row.get("terminal_status")
        if status == SUCCESS:
            for metric in ("ap70", "latency_ms", "energy_j"):
                _finite_metric(row, metric)
            if not _has_sha(row, "checkpoint_sha256"):
                raise ValueError("screen success is missing checkpoint SHA")
            if tuple(int(value) for value in row.get("width") or ()) != BASE_WIDTH:
                if not _has_sha(row, "recovery_training_report_sha256"):
                    raise ValueError(
                        "pruned screen success is missing recovery-training SHA"
                    )
        elif status in TERMINAL_FAILURES:
            if not str(row.get("failure_reason") or "").strip():
                raise ValueError("screen terminal failure is missing failure_reason")
        else:
            raise ValueError("screen feedback contains a non-terminal row")
    successful = [row for row in rows if row["terminal_status"] == SUCCESS]
    if len(successful) < 4:
        raise ValueError("fewer than four successful screen rows can be tuned")
    floor = float(ap70_ref) - float(max_ap_drop)
    feasible = [row for row in successful if float(row["ap70"]) >= floor]
    pool = feasible if len(feasible) >= 4 else successful
    return sorted(
        pool,
        key=lambda row: (
            float(row["latency_ms"]),
            float(row["energy_j"]),
            -float(row["ap70"]),
            str(row["row_id"]),
        ),
    )[:4]


def fixed_original_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    explicit = [
        dict(row)
        for row in candidates
        if tuple(int(value) for value in row.get("width") or ()) == BASE_WIDTH
        and row.get("q_mode") == "fp32"
    ]
    if len(explicit) == 1:
        return explicit[0]
    if explicit:
        raise ValueError(
            "schedule-only requires exactly one scanner-derived original FP32 row"
        )
    base_rows = [
        dict(row)
        for row in candidates
        if tuple(int(value) for value in row.get("width") or ()) == BASE_WIDTH
        and row.get("q_mode") in {"fp16", "int8"}
    ]
    groups = {
        (
            str(row.get("group_id") or ""),
            str(row.get("source_evidence_sha256") or ""),
        )
        for row in base_rows
    }
    if (
        {str(row.get("q_mode")) for row in base_rows} != {"fp16", "int8"}
        or len(groups) != 1
        or any(not value for value in next(iter(groups), ("", "")))
    ):
        raise ValueError(
            "schedule-only cannot identify one scanner-derived original structure"
        )
    source = min(base_rows, key=lambda row: str(row["q_mode"]))
    identity = hashlib.sha256(
        json.dumps(
            {
                "task_id": TASK_ID,
                "group_id": source["group_id"],
                "width": list(BASE_WIDTH),
                "q_mode": "fp32",
                "method": "schedule_only",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    row_id = f"S6-FCO-SCHEDULE-FP32-{identity[:20]}"
    return {
        **source,
        "row_id": row_id,
        "manifest_job_id": row_id,
        "q_mode": "fp32",
        "schedule_baseline_derivation": (
            "scanner_unique_original_structure_to_fixed_fp32"
        ),
        "schedule_baseline_derivation_sha256": identity,
    }


def _write_candidate_plan(
    *,
    output_dir: Path,
    arm_id: str,
    phase: str,
    rows: Sequence[Mapping[str, Any]],
    builder_level: int,
) -> None:
    selected = [dict(row) for row in rows]
    phase_root = output_dir / arm_id / phase
    _write(
        phase_root / "candidate_plan.json",
        {
            "schema_version": "stage6_fcooper_arm_candidate_plan_v2",
            "task_id": TASK_ID,
            "arm_id": arm_id,
            "phase": phase,
            "builder_optimization_level": builder_level,
            "row_count": len(selected),
            "rows": selected,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registry-json", type=Path, required=True)
    parser.add_argument("--observed-graphs-json", type=Path, required=True)
    parser.add_argument("--coldstart-rows-json", type=Path, required=True)
    parser.add_argument("--coldstart-graphs-json", type=Path, required=True)
    parser.add_argument("--profiles-json", type=Path, required=True)
    parser.add_argument("--frozen-contract-json", type=Path, required=True)
    parser.add_argument("--compress-then-tune-screen-feedback-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in vars(args).values():
        if isinstance(path, Path):
            reject_pilot_reference(str(path), context="input/output path")

    registry = _read_json(args.source_registry_json)
    profiles_payload = _read_json(args.profiles_json)
    profiles = _rows(profiles_payload, "capability_profiles")
    profile = next(row for row in profiles if row["dispatch_key"] == "trt_engine")
    candidates = candidates_from_scanner_registry(registry, profile)
    observed = _rows(_read_json(args.observed_graphs_json), "graph_features")
    reject_pilot_reference(observed, context="observed graph evidence")
    graph_costs = fit_graph_costs(observed, candidates)
    contract = _read_json(args.frozen_contract_json)
    reject_pilot_reference(contract, context="frozen contract")
    original_graph = next(
        (
            row
            for row in observed
            if tuple(int(value) for value in row.get("width") or ()) == BASE_WIDTH
        ),
        dict(contract.get("original_graph_features") or {}),
    )
    coldstart_rows, coldstart_graphs, coldstart_audit = load_frozen_gold176(
        args.coldstart_rows_json, args.coldstart_graphs_json
    )
    ap_by_genome = fit_neutral_ap(
        coldstart_rows,
        coldstart_graphs,
        candidates,
        fcooper_ap70_ref=float(contract["ap70_ref"]),
        fcooper_original_graph=original_graph,
    )
    ranked = rank_backend_neutral_candidates(candidates, ap_by_genome, graph_costs)
    protocol = build_five_arm_protocol(ranked, candidate_pool_size=len(candidates))
    protocol["coldstart_contract"] = {
        "source": "byte_frozen_standalone_gold176_only",
        **coldstart_audit,
    }
    protocol["observed_graph_evidence_contract"] = {
        "path": str(args.observed_graphs_json.resolve()),
        "sha256": _file_sha(args.observed_graphs_json),
        "t16_actual_graph_count": 16,
        "base_graph_count": 1,
    }
    by_id = {str(row["row_id"]): row for row in candidates}
    original = fixed_original_candidate(candidates)
    protocol["arms"]["schedule_only"].update(
        {
            "fixed_row_id": original["row_id"],
            "fixed_manifest_job_id": original["manifest_job_id"],
            "schedule_baseline_derivation": original[
                "schedule_baseline_derivation"
            ],
            "schedule_baseline_derivation_sha256": original[
                "schedule_baseline_derivation_sha256"
            ],
        }
    )

    compression_ids = protocol["arms"]["compression_only"]["selected_row_ids"]
    _write_candidate_plan(
        output_dir=args.output_dir,
        arm_id="compression_only",
        phase="measure",
        rows=[by_id[row_id] for row_id in compression_ids],
        builder_level=0,
    )
    _write_candidate_plan(
        output_dir=args.output_dir,
        arm_id="schedule_only",
        phase="measure",
        rows=[original],
        builder_level=5,
    )
    tune_arm = protocol["arms"]["compress_then_tune"]
    screen_ids = tune_arm["screen_row_ids"]
    _write_candidate_plan(
        output_dir=args.output_dir,
        arm_id="compress_then_tune",
        phase="screen",
        rows=[by_id[row_id] for row_id in screen_ids],
        builder_level=0,
    )
    if args.compress_then_tune_screen_feedback_json is not None:
        feedback_payload = _read_json(args.compress_then_tune_screen_feedback_json)
        reject_pilot_reference(feedback_payload, context="screen feedback")
        tuned_feedback = select_tuned_remeasurements(
            _rows(feedback_payload),
            screen_row_ids=screen_ids,
            ap70_ref=float(contract["ap70_ref"]),
            max_ap_drop=0.10,
        )
        tuned_ids = [str(row["row_id"]) for row in tuned_feedback]
        tune_arm["tuned_row_ids"] = tuned_ids
        tune_arm["phase_status"] = "tuned_remeasurement_ready"
        tune_arm["screen_feedback_path"] = str(
            args.compress_then_tune_screen_feedback_json.resolve()
        )
        _write_candidate_plan(
            output_dir=args.output_dir,
            arm_id="compress_then_tune",
            phase="tuned_remeasurement",
            rows=[by_id[row_id] for row_id in tuned_ids],
            builder_level=5,
        )
    _write(args.output_dir / "stage6_fcooper_five_arm_plan.json", protocol)
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "candidate_pool": len(candidates),
                "compression_only": 16,
                "compress_then_tune_screen": 12,
                "compress_then_tune_tuned": len(tune_arm["tuned_row_ids"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
