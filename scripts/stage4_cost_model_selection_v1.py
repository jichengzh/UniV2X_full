#!/usr/bin/env python3
"""Run the first Gold176 nested grouped cost-model/loss comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage4.cost_model_selection_v1 import (
    DEFAULT_CANDIDATES,
    run_nested_selection,
)


DEFAULT_GOLD = (
    REPO_ROOT
    / "results/stage35_gold144_targeted_supplement_v2_20260714"
    / "final_gold176_v1/gold176_final.json"
)
DEFAULT_GRAPH = DEFAULT_GOLD.with_name("graph_features.json")
DEFAULT_PROFILES = REPO_ROOT / "results/s1_profile_final_v3_20260711/capability_profiles_v3.json"
DEFAULT_OUTPUT = REPO_ROOT / "results/stage4_cost_model_selection_v1_20260716"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(payload: Any, field: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, Mapping) and isinstance(payload.get(field), list):
        return [dict(item) for item in payload[field]]
    raise ValueError(f"expected a list or object containing {field}")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-json", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--graph-features-json", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--capability-profiles-json", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = tuple(item.strip() for item in args.candidates.split(",") if item.strip())
    if not candidates:
        raise ValueError("--candidates must not be empty")
    rows = _rows(json.loads(args.gold_json.read_text(encoding="utf-8")), "rows")
    graph = _rows(
        json.loads(args.graph_features_json.read_text(encoding="utf-8")),
        "graph_features",
    )
    profiles = _rows(
        json.loads(args.capability_profiles_json.read_text(encoding="utf-8")),
        "capability_profiles",
    )
    report = run_nested_selection(
        rows,
        graph,
        profiles,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        seed=args.seed,
        candidates=candidates,
    )
    report = {
        **report,
        "inputs": {
            "gold_json": str(args.gold_json),
            "gold_sha256": _sha256(args.gold_json),
            "graph_features_json": str(args.graph_features_json),
            "graph_features_sha256": _sha256(args.graph_features_json),
            "capability_profiles_json": str(args.capability_profiles_json),
            "capability_profiles_sha256": _sha256(args.capability_profiles_json),
        },
        "candidate_names": list(candidates),
    }
    _write_json_atomic(args.output_dir / "stage4_cost_model_selection_report.json", report)

    summary_rows = []
    fold_rows = []
    for target, payload in report["targets"].items():
        summary_rows.append(
            {
                "target": target,
                **payload["summary"],
                "selected_candidate_counts": json.dumps(
                    payload["selected_candidate_counts"], sort_keys=True
                ),
            }
        )
        for fold in payload["outer_folds"]:
            fold_rows.append(
                {
                    "target": target,
                    "outer_fold": fold["outer_fold"],
                    "selected_candidate": fold["selected_candidate"],
                    "train_groups": len(fold["train_groups"]),
                    "test_groups": len(fold["test_groups"]),
                    "train_rows": fold["train_rows"],
                    "test_rows": fold["test_rows"],
                    **fold["metrics"],
                }
            )
    _write_csv(
        args.output_dir / "model_family_summary.csv",
        summary_rows,
        ["target", "mae", "mape", "spearman", "selected_candidate_counts"],
    )
    _write_csv(
        args.output_dir / "nested_cv_folds.csv",
        fold_rows,
        [
            "target",
            "outer_fold",
            "selected_candidate",
            "train_groups",
            "test_groups",
            "train_rows",
            "test_rows",
            "mae",
            "mape",
            "spearman",
        ],
    )
    print(json.dumps({"output_dir": str(args.output_dir), "summary": summary_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
