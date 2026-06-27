#!/usr/bin/env python3
"""Generate Stage2 coverage-first FP16 Pyramid backbone candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import utc_timestamp  # noqa: E402


CANDIDATE_SCHEMA = "stage2_candidate_row_v1"
DEFAULT_MODEL = "Pyramid-LiDAR"
DEFAULT_LIMIT = 60
S0_WIDTHS = [16, 24, 32, 40, 48, 56, 64]
S1_WIDTHS = [32, 48, 64, 80, 96, 112, 128]
S2_WIDTHS = [64, 96, 128, 160, 192, 224, 256]
BASE_WIDTH = [64, 128, 256]
REQUIRED_AXES = ["latency", "energy", "ap"]
AP_POLICY = "true_eval_or_true_import_only"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            validate_candidate_row(row)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def validate_candidate_row(row: dict[str, Any]) -> None:
    required = {
        "schema",
        "candidate_id",
        "model",
        "label",
        "width",
        "arm",
        "quant_policy",
        "schedule_policy",
        "optimized_scope",
        "priority",
        "axes_required",
        "repeat_policy",
        "max_latency_repeats",
        "max_energy_repeats",
        "ap_policy",
        "created_at",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"candidate row missing fields: {sorted(missing)}")
    if row["schema"] != CANDIDATE_SCHEMA:
        raise ValueError(f"unexpected candidate schema: {row['schema']}")
    width = row["width"]
    if not isinstance(width, list) or len(width) != 3:
        raise ValueError("candidate width must be a three-item list")
    if not all(isinstance(item, int) and item > 0 for item in width):
        raise ValueError("candidate width values must be positive integers")
    if row["quant_policy"] != "fp16":
        raise ValueError("Stage2 coverage candidate generator is FP16-only")
    if row["optimized_scope"] != "backbone_only":
        raise ValueError("Stage2 coverage candidates must be backbone_only")
    if row["axes_required"] != REQUIRED_AXES:
        raise ValueError("coverage candidates must request latency, energy, and ap axes")
    if int(row["max_latency_repeats"]) != 1 or int(row["max_energy_repeats"]) != 1:
        raise ValueError("coverage candidates cap latency and energy repeats at 1")


def config_id_for_width(width: list[int], schedule_policy: str) -> str:
    return (
        f"coverage_h800_tvm_pyramid_w{width[0]}x{width[1]}x{width[2]}"
        f"_fp16_{schedule_policy}"
    )


def software_point_id_for_width(width: list[int]) -> str:
    return f"pyramid_lidar:backbone:w{width[0]}x{width[1]}x{width[2]}:fp16"


def candidate_id_for_width(width: list[int]) -> str:
    return f"coverage:pyramid_lidar:w{width[0]}x{width[1]}x{width[2]}:fp16"


def _width_tuple(row: dict[str, Any]) -> tuple[int, int, int] | None:
    width = row.get("width")
    if isinstance(width, list) and len(width) == 3:
        try:
            return (int(width[0]), int(width[1]), int(width[2]))
        except (TypeError, ValueError):
            return None
    return None


def _existing_keys(rows: Iterable[dict[str, Any]]) -> dict[str, set[Any]]:
    labels: set[str] = set()
    config_ids: set[str] = set()
    candidate_ids: set[str] = set()
    widths: set[tuple[int, int, int]] = set()
    for row in rows:
        label = row.get("label")
        if label:
            labels.add(str(label))
        candidate_id = row.get("candidate_id")
        if candidate_id:
            text = str(candidate_id)
            candidate_ids.add(text)
            tail = text.rsplit(":", 1)[-1]
            if tail and not tail.startswith("fp16"):
                labels.add(tail)
        for key in ("config_id", "config_id_default", "config_id_tuned"):
            value = row.get(key)
            if value:
                config_ids.add(str(value))
        for value in row.get("config_ids", []) or []:
            if value:
                config_ids.add(str(value))
        width = _width_tuple(row)
        if width is not None:
            widths.add(width)
    return {
        "labels": labels,
        "config_ids": config_ids,
        "candidate_ids": candidate_ids,
        "widths": widths,
    }


def _candidate_row(
    *,
    label: str,
    width: tuple[int, int, int],
    arm: str,
    priority: int,
    sample_source: str,
    model: str,
    created_at: str,
    schedule_policy: str,
    optimized_scope: str,
) -> dict[str, Any]:
    width_list = [int(width[0]), int(width[1]), int(width[2])]
    config_default = config_id_for_width(width_list, "default")
    config_tuned = config_id_for_width(width_list, "metaschedule_tuned")
    return {
        "schema": CANDIDATE_SCHEMA,
        "candidate_id": candidate_id_for_width(width_list),
        "model": model,
        "label": label,
        "width": width_list,
        "arm": arm,
        "quant_policy": "fp16",
        "schedule_policy": schedule_policy,
        "optimized_scope": optimized_scope,
        "priority": int(priority),
        "axes_required": list(REQUIRED_AXES),
        "repeat_policy": "coverage",
        "max_latency_repeats": 1,
        "max_energy_repeats": 1,
        "ap_policy": AP_POLICY,
        "created_at": created_at,
        "dense_stage": "backbone",
        "software_point_id": software_point_id_for_width(width_list),
        "config_id_default": config_default,
        "config_id_tuned": config_tuned,
        "config_ids": [config_default, config_tuned],
        "sample_source": sample_source,
    }


def _single_axis_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for width in S0_WIDTHS:
        specs.append(
            {
                "label": f"s0_{width:03d}",
                "width": (width, BASE_WIDTH[1], BASE_WIDTH[2]),
                "arm": "P",
                "sample_source": "single_axis_s0",
                "priority": 100 - len(specs),
            }
        )
    for width in S1_WIDTHS:
        specs.append(
            {
                "label": f"s1_{width:03d}",
                "width": (BASE_WIDTH[0], width, BASE_WIDTH[2]),
                "arm": "P",
                "sample_source": "single_axis_s1",
                "priority": 100 - len(specs),
            }
        )
    for width in S2_WIDTHS:
        specs.append(
            {
                "label": f"s2_{width:03d}",
                "width": (BASE_WIDTH[0], BASE_WIDTH[1], width),
                "arm": "P",
                "sample_source": "single_axis_s2",
                "priority": 100 - len(specs),
            }
        )
    return specs


def _coupled_latin_hypercube_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for cycle in range(12):
        for idx in range(7):
            width = (
                S0_WIDTHS[(idx + cycle) % len(S0_WIDTHS)],
                S1_WIDTHS[(2 * idx + cycle) % len(S1_WIDTHS)],
                S2_WIDTHS[(3 * idx + 2 * cycle) % len(S2_WIDTHS)],
            )
            if width in seen:
                continue
            seen.add(width)
            specs.append(
                {
                    "label": f"lhc_{len(specs):02d}",
                    "width": width,
                    "arm": "P/S",
                    "sample_source": "coupled_latin_hypercube",
                    "priority": 80 - len(specs),
                }
            )
    return specs


def _nearest(value: int, choices: list[int]) -> int:
    return min(choices, key=lambda item: (abs(item - value), item))


def _frontier_neighborhood_specs() -> list[dict[str, Any]]:
    seeds = [
        (32, 64, 128),
        (16, 32, 64),
        (48, 96, 192),
        (64, 96, 192),
        (48, 128, 256),
        (64, 96, 256),
        (40, 80, 160),
        (56, 112, 224),
    ]
    deltas = [
        (0, 0, 0),
        (-8, 0, 0),
        (8, 0, 0),
        (0, -16, 0),
        (0, 16, 0),
        (0, 0, -32),
        (0, 0, 32),
        (-8, -16, -32),
        (8, 16, 32),
    ]
    specs: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for seed in seeds:
        for delta in deltas:
            width = (
                _nearest(seed[0] + delta[0], S0_WIDTHS),
                _nearest(seed[1] + delta[1], S1_WIDTHS),
                _nearest(seed[2] + delta[2], S2_WIDTHS),
            )
            if width in seen:
                continue
            seen.add(width)
            specs.append(
                {
                    "label": f"frontier_{len(specs):02d}",
                    "width": width,
                    "arm": "P/S",
                    "sample_source": "frontier_neighborhood",
                    "priority": 60 - len(specs),
                }
            )
    return specs


def _add_specs(
    *,
    rows: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    target_count: int,
    existing: dict[str, set[Any]],
    seen: dict[str, set[Any]],
    model: str,
    created_at: str,
    schedule_policy: str,
    optimized_scope: str,
) -> None:
    for spec in specs:
        if len(rows) >= target_count:
            return
        width = tuple(spec["width"])
        width_list = [int(width[0]), int(width[1]), int(width[2])]
        candidate_id = candidate_id_for_width(width_list)
        config_default = config_id_for_width(width_list, "default")
        config_tuned = config_id_for_width(width_list, "metaschedule_tuned")
        label = str(spec["label"])
        if label in existing["labels"] or label in seen["labels"]:
            continue
        if candidate_id in existing["candidate_ids"] or candidate_id in seen["candidate_ids"]:
            continue
        if width in existing["widths"] or width in seen["widths"]:
            continue
        if config_default in existing["config_ids"] or config_tuned in existing["config_ids"]:
            continue
        row = _candidate_row(
            label=label,
            width=width,
            arm=str(spec["arm"]),
            priority=int(spec["priority"]),
            sample_source=str(spec["sample_source"]),
            model=model,
            created_at=created_at,
            schedule_policy=schedule_policy,
            optimized_scope=optimized_scope,
        )
        validate_candidate_row(row)
        rows.append(row)
        seen["labels"].add(label)
        seen["candidate_ids"].add(candidate_id)
        seen["widths"].add(width)
        seen["config_ids"].update({config_default, config_tuned})


def generate_coverage_candidates(
    *,
    limit: int = DEFAULT_LIMIT,
    existing_rows: list[dict[str, Any]] | None = None,
    created_at: str | None = None,
    model: str = DEFAULT_MODEL,
    schedule_policy: str = "metaschedule_tuned",
    optimized_scope: str = "backbone_only",
) -> list[dict[str, Any]]:
    if int(limit) <= 0:
        return []
    existing = _existing_keys(existing_rows or [])
    seen = {"labels": set(), "config_ids": set(), "candidate_ids": set(), "widths": set()}
    rows: list[dict[str, Any]] = []
    created = created_at or utc_timestamp()

    _add_specs(
        rows=rows,
        specs=_single_axis_specs(),
        target_count=limit,
        existing=existing,
        seen=seen,
        model=model,
        created_at=created,
        schedule_policy=schedule_policy,
        optimized_scope=optimized_scope,
    )
    coupled_target = min(limit, len(rows) + 25)
    _add_specs(
        rows=rows,
        specs=_coupled_latin_hypercube_specs(),
        target_count=coupled_target,
        existing=existing,
        seen=seen,
        model=model,
        created_at=created,
        schedule_policy=schedule_policy,
        optimized_scope=optimized_scope,
    )
    _add_specs(
        rows=rows,
        specs=_frontier_neighborhood_specs(),
        target_count=limit,
        existing=existing,
        seen=seen,
        model=model,
        created_at=created,
        schedule_policy=schedule_policy,
        optimized_scope=optimized_scope,
    )
    if len(rows) < limit:
        raise ValueError(f"only generated {len(rows)} unique candidates, requested {limit}")
    return rows[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--existing-jsonl", action="append", default=[])
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--created-at")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--schedule-policy", default="metaschedule_tuned")
    parser.add_argument("--optimized-scope", default="backbone_only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    existing_rows: list[dict[str, Any]] = []
    for path in args.existing_jsonl:
        existing_rows.extend(read_jsonl(path))
    rows = generate_coverage_candidates(
        limit=args.limit,
        existing_rows=existing_rows,
        created_at=args.created_at,
        model=args.model,
        schedule_policy=args.schedule_policy,
        optimized_scope=args.optimized_scope,
    )
    write_jsonl(args.out_jsonl, rows)
    sources: dict[str, int] = {}
    for row in rows:
        source = str(row["sample_source"])
        sources[source] = sources.get(source, 0) + 1
    print(
        json.dumps(
            {
                "schema": "stage2_coverage_candidate_generation_summary_v1",
                "out": args.out_jsonl,
                "candidates": len(rows),
                "sample_sources": sources,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
