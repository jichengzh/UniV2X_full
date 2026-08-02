#!/usr/bin/env python3
"""Freeze executable CoDriving Stage6 candidate and arm plans."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage6.formal_plan_v1 import fit_neutral_ap_surrogate  # noqa: E402
from framework.stage5.single_target_search_v2 import SearchTask, build_measurement_request  # noqa: E402


def _rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"expected candidate rows: {path}")
    return [dict(row) for row in rows]


def _key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    genome = list(row["genome"])
    return int(genome[0]), int(genome[1]), int(genome[2]), str(genome[3])


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize(values: list[float], value: float) -> float:
    low, high = min(values), max(values)
    return 0.0 if high == low else (value - low) / (high - low)


def _finite(value: Any) -> float:
    number = float(value)
    if not (number == number and abs(number) != float("inf")):
        raise ValueError(f"non-finite scalar: {value!r}")
    return number


def _q_bits(q_mode: str) -> int:
    if q_mode == "int8":
        return 8
    if q_mode == "fp16":
        return 16
    if q_mode == "fp32":
        return 32
    raise ValueError(f"unsupported q_mode for bit-cost plan: {q_mode}")


def _build_bitcost_formal_plan(
    tvm_rows: list[dict[str, Any]],
    trt_rows: list[dict[str, Any]],
    *,
    expected_pool_size: int,
    neutral_ap_by_genome: dict[tuple[int, int, int, str], float],
) -> dict[str, Any]:
    tvm, trt = {_key(row): row for row in tvm_rows}, {_key(row): row for row in trt_rows}
    if len(tvm) != len(tvm_rows) or len(trt) != len(trt_rows):
        raise ValueError("duplicate candidate genome in CoDriving formal candidates")
    if set(tvm) != set(trt):
        raise ValueError("TVM/TRT candidate genome sets differ")
    if len(tvm) != expected_pool_size:
        raise ValueError(f"effective candidate pool mismatch: expected {expected_pool_size}, got {len(tvm)}")

    neutral = []
    for key in sorted(tvm):
        left, right = tvm[key], trt[key]
        graph_left = left.get("graph_features") or {}
        graph_right = right.get("graph_features") or {}
        parameter_count = _finite(graph_left.get("parameter_elements"))
        flops = _finite(graph_left.get("conv_flops"))
        if abs(parameter_count - _finite(graph_right.get("parameter_elements"))) > 1e-6:
            raise ValueError(f"backend candidate parameter drift for genome {key}")
        if abs(flops - _finite(graph_right.get("conv_flops"))) > 1e-3:
            raise ValueError(f"backend candidate FLOPs drift for genome {key}")
        bits = _q_bits(key[3])
        neutral.append({
            "genome": [*key[:3], key[3]],
            "candidate_id": f"codriving|{'x'.join(map(str, key[:3]))}|q={key[3]}",
            "ap_surrogate": _finite(neutral_ap_by_genome[key]),
            "parameter_count": parameter_count,
            "flops": flops,
            "q_bits": bits,
            "parameter_bits": parameter_count * bits,
            "bitops": flops * bits,
            "source_evidence_sha256": left.get("source_evidence_sha256"),
        })

    columns = {
        name: [float(row[name]) for row in neutral]
        for name in ("ap_surrogate", "parameter_bits", "bitops")
    }
    ranked = []
    for row in neutral:
        score = (
            _normalize(columns["ap_surrogate"], float(row["ap_surrogate"]))
            - 0.5 * _normalize(columns["parameter_bits"], float(row["parameter_bits"]))
            - 0.5 * _normalize(columns["bitops"], float(row["bitops"]))
        )
        ranked.append({**row, "hardware_blind_acquisition_score": score})
    ranked.sort(key=lambda row: (-row["hardware_blind_acquisition_score"], row["candidate_id"]))

    selected = [row["genome"] for row in ranked[:16]]
    screened = [row["genome"] for row in ranked[:12]]
    locked = [row["genome"] for row in ranked[:4]]
    backend_plans = {
        backend: {
            "compression_only": [list(genome) for genome in selected],
            "compress_then_tune": [list(genome) for genome in locked],
            "tune_then_compress": [list(genome) for genome in selected],
        }
        for backend in ("tvm", "trt")
    }
    payload = {
        "schema_version": "stage6_formal_execution_plan_v1",
        "passed": True,
        "effective_candidate_pool_size": len(neutral),
        "pool_policy": "registered_materializable_minus_frozen_gold176_rows",
        "hardware_blind_backend_labels_used": False,
        "hardware_blind_cost_policy": "parameter_bits_and_bitops",
        "ranked_candidates": ranked,
        "arms": {
            "original_default": {"fixed_genome": [64, 128, 256, "fp32"]},
            "compression_only": {"selected_genomes": selected, "outer_budget": 16},
            "schedule_only": {"fixed_genome": [64, 128, 256, "fp32"]},
            "compress_then_tune": {
                "screened_genomes": screened,
                "locked_genomes": locked,
                "outer_budget": {"screen": 12, "locked": 4},
            },
            "tune_then_compress": {
                "attempt_genomes": selected,
                "outer_budget": {"base_tune": 1, "compressed_attempts": 16},
            },
            "joint_shcosearch": {"reuse_actual_feedback_v3_budget": 16},
        },
        "backend_plans": backend_plans,
    }
    return {**payload, "plan_sha256": _digest(payload)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = ROOT / "results/stage5_codriving_actual_v3_20260721"
    parser.add_argument("--tvm-candidates", type=Path, default=default_root / "S5-COD-TVM/round_00/predicted_candidates.json")
    parser.add_argument("--trt-candidates", type=Path, default=default_root / "S5-COD-TRT/round_00/predicted_candidates.json")
    parser.add_argument("--expected-pool-size", type=int, default=648)
    coldstart = ROOT / "results/stage35_gold144_targeted_supplement_v2_20260714/final_gold176_v1"
    parser.add_argument("--coldstart-rows", type=Path, default=coldstart / "gold176_final.json")
    parser.add_argument("--coldstart-graphs", type=Path, default=coldstart / "graph_features.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profiles-json", type=Path, default=ROOT / "results/s1_profile_final_v3_20260711/capability_profiles_v3.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backend_rows = {"tvm": _rows(args.tvm_candidates), "trt": _rows(args.trt_candidates)}
    neutral_ap = fit_neutral_ap_surrogate(
        _rows(args.coldstart_rows),
        _rows(args.coldstart_graphs),
        backend_rows["tvm"],
    )
    plan = _build_bitcost_formal_plan(
        backend_rows["tvm"],
        backend_rows["trt"],
        expected_pool_size=args.expected_pool_size,
        neutral_ap_by_genome=neutral_ap,
    )
    plan = {
        **plan,
        "experiment_id": "stage6-codriving-h800-six-arm-v1",
        "target_model": "codriving",
        "note": "candidate_id values inside ranked_candidates may inherit the generic formal_plan label; executable rows and measurement requests retain codriving identities.",
    }
    _write(args.output_dir / "stage6_formal_execution_plan_v1.json", plan)

    profiles_payload = json.loads(args.profiles_json.read_text(encoding="utf-8"))
    profiles = profiles_payload.get("capability_profiles") if isinstance(profiles_payload, dict) else profiles_payload
    for backend, rows in backend_rows.items():
        indexed = {_key(row): row for row in rows}
        dispatch_key = "tvm_auto" if backend == "tvm" else "trt_engine"
        profile = next(row for row in profiles if row["dispatch_key"] == dispatch_key)
        task = SearchTask(f"S5-COD-{backend.upper()}", "codriving", "h800", profile)
        for arm in ("compression_only", "compress_then_tune", "tune_then_compress"):
            selected = plan["backend_plans"][backend][arm]
            payload = {
                "schema_version": "stage6_formal_arm_candidate_plan_v1",
                "plan_sha256": plan["plan_sha256"],
                "backend": backend,
                "arm_id": arm,
                "target_model": "codriving",
                "candidate_count": len(selected),
                "rows": [indexed[tuple(genome)] for genome in selected],
            }
            _write(args.output_dir / backend / f"{arm}_candidate_plan.json", payload)
            if arm in {"compression_only", "compress_then_tune"}:
                for batch_index in range(0, len(selected), 4):
                    batch_rows = [indexed[tuple(genome)] for genome in selected[batch_index : batch_index + 4]]
                    request = build_measurement_request(task=task, selected_rows=batch_rows, round_index=batch_index // 4)
                    _write(args.output_dir / backend / arm / f"batch_{batch_index // 4:02d}" / "measurement_request.json", request)

    runner = ROOT / "scripts/stage6_formal_runner_v1.py"
    runner_plan = {
        "schema_version": "stage6_formal_runner_plan_v1",
        "experiment_id": "stage6-codriving-h800-six-arm-v1",
        "target_model": "codriving",
        "hardware_id": "h800",
        "backends": ["tvm", "trt"],
        "arm_ids": ["original_default", "compression_only", "schedule_only", "compress_then_tune", "tune_then_compress", "joint_shcosearch"],
        "formal_execution_plan": str((args.output_dir / "stage6_formal_execution_plan_v1.json").resolve()),
        "formal_execution_plan_sha256": _sha(args.output_dir / "stage6_formal_execution_plan_v1.json"),
        "formal_runner_path": str(runner.resolve()),
        "formal_runner_sha256": _sha(runner),
        "output_root": str(args.output_dir.resolve()),
        "budget_contract_bound": True,
    }
    _write(args.output_dir / "stage6_formal_runner_plan_v1.json", runner_plan)
    print(json.dumps({"passed": True, "plan_sha256": plan["plan_sha256"], "candidate_pool": args.expected_pool_size}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
