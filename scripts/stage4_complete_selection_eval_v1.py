#!/usr/bin/env python3
"""Run Stage4 ranking, uncertainty, Pareto/HV, and offline replay evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage4.selection_completion_v1 import run_stage4_completion


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sequence(path: Path, keys: Sequence[str]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict):
        for key in keys:
            if isinstance(payload.get(key), list):
                return [dict(item) for item in payload[key]]
    raise ValueError(f"{path} does not contain a supported row list")


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("--replay-seeds must not be empty")
    return seeds


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-json", type=Path, required=True)
    parser.add_argument("--graph-features-json", type=Path, required=True)
    parser.add_argument("--capability-profiles-json", type=Path, required=True)
    parser.add_argument("--cost-model-report-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--uncertainty-splits", type=int, default=5)
    parser.add_argument("--replay-initial-groups", type=int, default=6)
    parser.add_argument("--replay-budget-groups", type=int)
    parser.add_argument("--replay-seeds", default="0,1,2,3,4")
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _load_sequence(args.gold_json, ("rows", "jobs"))
    graph_features = _load_sequence(args.graph_features_json, ("rows", "features"))
    capability_profiles = _load_sequence(args.capability_profiles_json, ("profiles",))
    cost_report = json.loads(args.cost_model_report_json.read_text(encoding="utf-8"))
    report = run_stage4_completion(
        rows,
        graph_features,
        capability_profiles,
        cost_report,
        outer_splits=args.outer_splits,
        uncertainty_splits=args.uncertainty_splits,
        replay_initial_groups=args.replay_initial_groups,
        replay_budget_groups=args.replay_budget_groups,
        replay_seeds=_parse_seeds(args.replay_seeds),
        seed=args.seed,
    )
    input_sha256 = {
        "gold": _sha256(args.gold_json),
        "graph_features": _sha256(args.graph_features_json),
        "capability_profiles": _sha256(args.capability_profiles_json),
        "cost_model_report": _sha256(args.cost_model_report_json),
    }
    report["input_sha256"] = input_sha256
    summary = {
        **report["summary"],
        "schema_version": "stage4_completion_summary_v1",
        "input_sha256": input_sha256,
        "evaluation_audit": report["evaluation_audit"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "stage4_ranking_pareto_report.json": report["ranking_pareto"],
        "stage4_uncertainty_report.json": report["uncertainty"],
        "stage4_closed_loop_replay_report.json": report["replay"],
        "stage4_completion_summary.json": summary,
        "stage4_completion_report.json": report,
    }
    for name, payload in outputs.items():
        _write_json(args.output_dir / name, payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
