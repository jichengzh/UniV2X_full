#!/usr/bin/env python3
"""Evaluate Feedback16 as a true post-coldstart incremental update."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage4.feedback_update_eval_v1 import run_feedback_update_evaluation


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-json", type=Path, required=True)
    parser.add_argument("--graph-features-json", type=Path, required=True)
    parser.add_argument("--capability-profiles-json", type=Path, required=True)
    parser.add_argument("--cold-cost-report-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    report = run_feedback_update_evaluation(
        _load(args.rows_json),
        _load(args.graph_features_json),
        _load(args.capability_profiles_json),
        _load(args.cold_cost_report_json),
        seed=args.seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {
        mode: {
            "simultaneous_group_coverage": report[mode]["simultaneous_group_coverage"],
            "targets": {
                target: report[mode]["targets"][target]
                for target in ("latency_ms", "energy_j", "ap70")
            },
        }
        for mode in ("before_feedback", "after_feedback")
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
