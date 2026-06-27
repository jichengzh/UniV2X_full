#!/usr/bin/env python3
"""Generate Safe Stage1 coupling predictions from partition manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage1.coupling_predictor import markdown_report, predict_manifests


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Stage1 partition YAML path. Repeatable.",
    )
    parser.add_argument(
        "--evidence-dir",
        default=str(ROOT / "results/stage1_model_predict"),
        help="Directory containing stage1_model_predict evidence JSON files.",
    )
    parser.add_argument("--out-json", required=True, help="Output JSON path.")
    parser.add_argument("--out-md", help="Optional Markdown report path.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = predict_manifests(args.manifest, evidence_dir=args.evidence_dir)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown_report(report), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
