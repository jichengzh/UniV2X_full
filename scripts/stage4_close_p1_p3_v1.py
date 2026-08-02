#!/usr/bin/env python3
"""Generate the fail-closed Stage4 P1-P3 closure audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage4.closure_audit_v1 import build_stage4_closure_audit


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold-cost-report", type=Path, required=True)
    parser.add_argument("--baseline-completion", type=Path, required=True)
    parser.add_argument("--feedback-completion", type=Path, required=True)
    parser.add_argument("--feedback-update", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--reproducibility-audit", type=Path, required=True)
    parser.add_argument("--merged-rows", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    inputs = {
        "cold_cost_report": args.cold_cost_report,
        "baseline_completion": args.baseline_completion,
        "feedback_completion": args.feedback_completion,
        "feedback_update": args.feedback_update,
        "holdout_manifest": args.holdout_manifest,
        "reproducibility_audit": args.reproducibility_audit,
        "merged_rows": args.merged_rows,
    }
    report = build_stage4_closure_audit(
        _load(args.cold_cost_report),
        _load(args.baseline_completion),
        _load(args.feedback_completion),
        _load(args.feedback_update),
        _load(args.holdout_manifest),
        reproducibility_audit=_load(args.reproducibility_audit),
        merged_rows=_load(args.merged_rows),
    )
    report["input_sha256"] = {name: _sha(path) for name, path in inputs.items()}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
